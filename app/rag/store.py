"""Vector store contract and the shared chunk type.

Chroma is behind this interface on purpose: its Python API has changed shape
several times across releases, and a future move to pgvector or LanceDB should
cost one file, not a refactor of the retriever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Chunk:
    """One retrievable passage plus the metadata we filter and boost on."""

    id: str
    text: str
    title: str = ""
    source: str = ""
    map: str = ""
    side: str = ""
    site: str = ""
    topic: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    rank_band: list[str] = field(default_factory=list)
    playstyle: list[str] = field(default_factory=list)
    score: float = 0.0

    # --- Chroma only accepts scalar metadata values, so lists round-trip as
    # --- pipe-joined strings. These two helpers are the only place that knows.

    LIST_FIELDS = ("topic", "agents", "roles", "rank_band", "playstyle")

    def to_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "title": self.title,
            "source": self.source,
            "map": self.map,
            "side": self.side,
            "site": self.site,
        }
        for f in self.LIST_FIELDS:
            meta[f] = "|".join(getattr(self, f))
        return meta

    @classmethod
    def from_metadata(cls, cid: str, text: str, meta: dict[str, Any]) -> Chunk:
        kwargs: dict[str, Any] = {
            "id": cid,
            "text": text,
            "title": meta.get("title", ""),
            "source": meta.get("source", ""),
            "map": meta.get("map", ""),
            "side": meta.get("side", ""),
            "site": meta.get("site", ""),
        }
        for f in cls.LIST_FIELDS:
            raw = meta.get(f) or ""
            kwargs[f] = [x for x in raw.split("|") if x]
        return cls(**kwargs)


class VectorStore(Protocol):
    def reset(self) -> None: ...

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def query(
        self, embedding: list[float], k: int, where: dict | None = None
    ) -> list[Chunk]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def count(self) -> int: ...
