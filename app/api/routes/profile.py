"""User profile CRUD."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.core.vocab import AGENTS_BY_ROLE, MAPS
from app.db.repositories import ProfileRepo
from app.schemas.profile import (
    PLAYSTYLES,
    RANKS,
    ROLES,
    STRENGTHS,
    WEAKNESSES,
    ProfileIn,
    ProfileOut,
)

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/options")
def options() -> dict[str, object]:
    """Valid values, so a UI never has to hardcode them."""
    return {
        "ranks": RANKS,
        "roles": ROLES,
        "playstyles": PLAYSTYLES,
        "agents_by_role": AGENTS_BY_ROLE,
        "maps": MAPS,
        "strengths": STRENGTHS,
        "weaknesses": WEAKNESSES,
    }


@router.get("/{username}", response_model=ProfileOut)
def get_profile(
    username: str, session: Session = Depends(get_session)
) -> ProfileOut:
    return ProfileRepo(session).get(username)


@router.put("/{username}", response_model=ProfileOut)
def put_profile(
    username: str, data: ProfileIn, session: Session = Depends(get_session)
) -> ProfileOut:
    return ProfileRepo(session).upsert(username, data)
