"""API-facing profile schemas.

The ORM stores list fields as newline-joined text; the API exposes them as
real lists. The conversion lives here so neither the DB layer nor the route
handlers have to think about it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from app.core.vocab import normalize_agent, normalize_map

RANKS = [
    "unranked", "iron", "bronze", "silver", "gold",
    "platinum", "diamond", "ascendant", "immortal", "radiant",
]
ROLES = ["duelist", "controller", "initiator", "sentinel", "flex"]
PLAYSTYLES = ["aggressive", "passive", "balanced"]

# Suggestions for the UI pickers. Not enforced: the API still accepts any
# free-text value, so older profiles and API clients keep working.
STRENGTHS = [
    "aim", "crosshair placement", "spray control", "movement", "game sense",
    "positioning", "utility usage", "comms", "trading", "clutching",
    "entry fragging", "economy", "map knowledge", "rotations", "retakes",
    "post-plant", "lurking", "staying calm",
]
WEAKNESSES = [
    "overpeeking", "dry peeking", "crosshair placement", "spray control",
    "movement", "positioning", "utility usage", "comms", "trading",
    "clutching", "economy", "map knowledge", "rotations", "retakes",
    "post-plant", "tilting", "whiffing", "inconsistency",
]

Playstyle = Literal["aggressive", "passive", "balanced"]


def _norm_rank(v: str) -> str:
    v = (v or "").strip().lower()
    # Accept "gold 2" / "Gold III" and keep only the tier.
    v = v.split()[0] if v else ""
    return v


def _known(values: list[str], normalize: Callable[[str], str | None]) -> list[str]:
    # Unknown names are dropped rather than rejected: retrieval ignores them
    # anyway, and rejecting would break profiles saved before this check
    # existed. Order is kept so the first pick stays the main.
    out: list[str] = []
    for x in values:
        n = normalize(x)
        if n and n not in out:
            out.append(n)
    return out


class ProfileIn(BaseModel):
    current_rank: str = "unranked"
    target_rank: str = ""
    preferred_role: str = ""
    playstyle: Playstyle = "balanced"
    main_agents: list[str] = Field(default_factory=list)
    favorite_maps: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    notes: str = ""
    # Mouse setup; 0 means not set.
    dpi: int = Field(default=0, ge=0, le=32000)
    sensitivity: float = Field(default=0.0, ge=0, le=10)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def edpi(self) -> int:
        """DPI x in-game sensitivity: the one number that compares setups."""
        return round(self.dpi * self.sensitivity) if self.dpi and self.sensitivity else 0

    @field_validator("current_rank", "target_rank")
    @classmethod
    def _rank(cls, v: str) -> str:
        v = _norm_rank(v)
        if v and v not in RANKS:
            raise ValueError(f"rank must be one of {RANKS}")
        return v

    @field_validator("preferred_role")
    @classmethod
    def _role(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v and v not in ROLES:
            raise ValueError(f"preferred_role must be one of {ROLES}")
        return v

    @field_validator("main_agents", mode="after")
    @classmethod
    def _agents(cls, v: list[str]) -> list[str]:
        return _known(v, normalize_agent)

    @field_validator("strengths", "weaknesses", mode="after")
    @classmethod
    def _skills(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for x in v:
            n = " ".join((x or "").strip().lower().split())
            if n and n not in out:
                out.append(n)
        return out

    @field_validator("favorite_maps", mode="after")
    @classmethod
    def _maps(cls, v: list[str]) -> list[str]:
        return _known(v, normalize_map)


class ProfileOut(ProfileIn):
    model_config = ConfigDict(from_attributes=True)

    user_id: str
    username: str
