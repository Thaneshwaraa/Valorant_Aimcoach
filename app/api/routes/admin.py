"""Health and knowledge-base administration."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_llm, get_retriever
from app.config import settings
from app.llm.ollama_client import OllamaClient
from app.rag.retriever import Retriever

router = APIRouter(tags=["admin"])


@router.get("/health")
async def health(
    llm: OllamaClient = Depends(get_llm),
    retriever: Retriever = Depends(get_retriever),
) -> dict:
    return {
        "status": "ok",
        "llm": {
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "reachable": await llm.health(),
        },
        "rag": {
            "collection": settings.chroma_collection,
            "chunks": retriever.store.count(),
        },
    }


@router.post("/kb/reindex")
def reindex(retriever: Retriever = Depends(get_retriever)) -> dict:
    """Rebuild the vector store from data/kb and drop the BM25 cache."""
    from app.rag.ingest import ingest

    count = ingest(reset=True)
    retriever.invalidate()
    return {"chunks": count}
