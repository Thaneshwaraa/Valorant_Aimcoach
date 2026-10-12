"""Text embeddings via fastembed (ONNX, CPU).

Deliberately CPU-only. The GPU has 6 GB and the LLM needs all of it; bge-small
embeds a query in ~15 ms on an i7, which is invisible inside the latency
budget. Keeping embeddings off the GPU is what lets the LLM stay resident.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

from app.config import settings
from app.logging_setup import get_logger

log = get_logger(__name__)


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embed_model
        log.info("embedder.load", model=self.model_name)
        self._model = TextEmbedding(model_name=self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models are trained with an instruction prefix on the query side;
        # using it lifts retrieval accuracy by a few points at zero cost.
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        return next(iter(self._model.query_embed([prefixed]))).tolist()


@lru_cache
def get_embedder() -> Embedder:
    """Process-wide singleton. Loading the ONNX model costs ~1 s."""
    return Embedder()
