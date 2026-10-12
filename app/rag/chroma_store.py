"""ChromaDB implementation of VectorStore.

Embeddings are always supplied by us, never computed by Chroma. That avoids
Chroma silently downloading its own default model and gives one consistent
embedding space between ingest and query time.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.logging_setup import get_logger
from app.rag.store import Chunk

log = get_logger(__name__)


class ChromaStore:
    def __init__(self, path: str | None = None, collection: str | None = None) -> None:
        self.path = str(settings.abs_path(path or settings.chroma_dir))
        self.collection_name = collection or settings.chroma_collection
        self._client = chromadb.PersistentClient(
            path=self.path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
            embeddings=embeddings,
        )

    def query(
        self, embedding: list[float], k: int, where: dict | None = None
    ) -> list[Chunk]:
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, max(self.count(), 1)),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        out: list[Chunk] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            chunk = Chunk.from_metadata(cid, doc, meta or {})
            # Cosine distance -> similarity, so bigger is always better.
            chunk.score = 1.0 - float(dist)
            out.append(chunk)
        return out

    def all_chunks(self) -> list[Chunk]:
        """Full dump, used to build the in-memory BM25 index at startup."""
        res = self._collection.get(include=["documents", "metadatas"])
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        return [
            Chunk.from_metadata(cid, doc, meta or {})
            for cid, doc, meta in zip(ids, docs, metas, strict=False)
        ]

    def count(self) -> int:
        return self._collection.count()
