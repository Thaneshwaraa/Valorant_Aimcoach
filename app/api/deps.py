"""Dependency wiring.

Single place where concrete implementations are chosen. Everything downstream
depends on protocols, so swapping Chroma for pgvector or Ollama for a hosted
API is an edit here and nowhere else.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.coach import CoachService
from app.db.session import get_session  # re-exported for routes
from app.llm.ollama_client import OllamaClient
from app.rag.chroma_store import ChromaStore
from app.rag.retriever import Retriever

__all__ = ["get_session", "get_llm", "get_retriever", "get_coach"]


@lru_cache
def get_llm() -> OllamaClient:
    return OllamaClient()


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(ChromaStore())


@lru_cache
def get_coach() -> CoachService:
    return CoachService(llm=get_llm(), retriever=get_retriever())
