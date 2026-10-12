"""Hybrid retrieval: dense vectors + BM25 + profile-aware re-ranking.

Why hybrid and not pure embeddings: this domain is almost entirely proper
nouns. "Heaven", "Catwalk", "A Main", "Chamber", "Killjoy" are what players
actually say, and dense vectors blur exactly those distinctions - a query
about Chamber will happily retrieve Killjoy chunks because both are sentinels.
BM25 anchors on the literal token, the embedder handles paraphrase, and
reciprocal-rank fusion combines them without needing the two score scales to
be comparable.
"""

from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

from app.config import settings
from app.core.filters import QueryContext
from app.logging_setup import get_logger
from app.rag.embedder import get_embedder
from app.rag.store import Chunk, VectorStore

log = get_logger(__name__)

RRF_K = 60  # standard reciprocal-rank-fusion damping constant

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Retriever:
    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._corpus: list[Chunk] = []

    # ------------------------------------------------------------------ BM25

    def _ensure_bm25(self) -> None:
        """Build the sparse index from whatever is in the vector store.

        Chroma is the single source of truth, so there is no second index to
        keep in sync on disk. A few hundred chunks build in well under 50 ms.
        """
        if self._bm25 is not None:
            return
        with self._lock:
            if self._bm25 is not None:
                return
            self._corpus = self.store.all_chunks()
            if not self._corpus:
                log.warning("bm25.empty_corpus")
                return
            self._bm25 = BM25Okapi([tokenize(c.text) for c in self._corpus])
            log.info("bm25.built", chunks=len(self._corpus))

    def invalidate(self) -> None:
        """Called after re-ingest so the next query rebuilds the index."""
        with self._lock:
            self._bm25 = None
            self._corpus = []

    # ------------------------------------------------------------- retrieval

    def _dense(self, ctx: QueryContext, n: int) -> list[Chunk]:
        embedding = get_embedder().embed_query(ctx.query)
        try:
            return self.store.query(embedding, k=n, where=ctx.where())
        except Exception as exc:  # noqa: BLE001 - bad filter must not 500
            log.warning("dense.filter_failed", error=str(exc))
            return self.store.query(embedding, k=n, where=None)

    def _sparse(self, ctx: QueryContext, n: int) -> list[Chunk]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(ctx.query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[Chunk] = []
        for i in ranked[: n * 3]:
            c = self._corpus[i]
            # Mirror the dense hard filter so fusion compares like with like.
            if not ctx.allows(c.map, c.side, c.site):
                continue
            c.score = float(scores[i])
            out.append(c)
            if len(out) >= n:
                break
        return out

    def _profile_boost(self, chunk: Chunk, ctx: QueryContext) -> float:
        """Small additive nudges. Deliberately never exclusionary.

        Magnitudes are tuned to be comparable to one or two RRF rank positions,
        so a strongly relevant chunk still outranks a weakly relevant one that
        happens to match the user's main agent.
        """
        boost = 0.0
        if ctx.agents and any(a in chunk.agents for a in ctx.agents):
            boost += 0.010
        if ctx.roles and any(r in chunk.roles for r in ctx.roles):
            boost += 0.004
        if ctx.rank_band and ctx.rank_band in chunk.rank_band:
            boost += 0.008
        if ctx.playstyle and ctx.playstyle in chunk.playstyle:
            boost += 0.004
        if ctx.side and chunk.side == ctx.side and (ctx.maps or not chunk.map):
            # "defense" only discriminates usefully inside a map context. When
            # the question names no map ("when should we rotate on defense?"),
            # boosting side would promote Ascent-specific notes over the
            # general rotations guide that actually answers it.
            boost += 0.008
        if ctx.site and chunk.site == ctx.site:
            boost += 0.006
        if ctx.maps and chunk.map in ctx.maps:
            boost += 0.010
        return boost

    @staticmethod
    def _diversify(ranked: list[Chunk], top_k: int, per_source: int = 4) -> list[Chunk]:
        """Cap how many chunks any single document may contribute.

        Without this, an on-the-nose question returns six chunks from one file
        and the answer has no access to the agent guide or the rank guide that
        make it personal. Two passes: take up to `per_source` from each
        document in rank order, then backfill from the leftovers so we always
        return top_k when the corpus allows.
        """
        picked: list[Chunk] = []
        overflow: list[Chunk] = []
        counts: dict[str, int] = {}
        for c in ranked:
            key = c.source or c.id
            if counts.get(key, 0) < per_source:
                counts[key] = counts.get(key, 0) + 1
                picked.append(c)
            else:
                overflow.append(c)
            if len(picked) >= top_k:
                return picked[:top_k]
        return (picked + overflow)[:top_k]

    def search(self, ctx: QueryContext, top_k: int | None = None) -> list[Chunk]:
        top_k = top_k or settings.retrieval_top_k
        n = settings.retrieval_candidates
        alpha = settings.hybrid_alpha

        dense = self._dense(ctx, n)
        sparse = self._sparse(ctx, n)

        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}

        for rank, c in enumerate(dense):
            fused[c.id] = fused.get(c.id, 0.0) + alpha / (RRF_K + rank + 1)
            by_id.setdefault(c.id, c)
        for rank, c in enumerate(sparse):
            fused[c.id] = fused.get(c.id, 0.0) + (1 - alpha) / (RRF_K + rank + 1)
            by_id.setdefault(c.id, c)

        for cid, base in fused.items():
            by_id[cid].score = base + self._profile_boost(by_id[cid], ctx)

        # Safety net: a chunk that slipped through either path but contradicts
        # the query (wrong map, wrong side) is worse than one fewer result.
        eligible = [c for c in by_id.values() if ctx.allows(c.map, c.side, c.site)]
        ranked = sorted(eligible, key=lambda c: c.score, reverse=True)
        results = self._diversify(ranked, top_k)
        log.info(
            "retrieval",
            dense=len(dense),
            sparse=len(sparse),
            returned=len(results),
            maps=ctx.maps,
            side=ctx.side,
            site=ctx.site,
            band=ctx.rank_band,
        )
        return results
