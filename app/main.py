"""FastAPI application.

The lifespan hook does the warm-up that the latency budget depends on: load
the embedding model, build the BM25 index, and poke Ollama so the LLM is
resident in VRAM before the first question arrives. Without this the first
request pays ~3 s of cold start.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import get_llm, get_retriever
from app.api.routes import admin, chat, profile, web
from app.config import settings
from app.db.models import Base
from app.db.session import engine
from app.logging_setup import configure_logging, get_logger
from app.rag.embedder import get_embedder

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info("startup", model=settings.llm_model, db=settings.database_url)

    # Alembic owns the schema in normal operation; this is a safety net so a
    # fresh clone works before the first `alembic upgrade head`.
    Base.metadata.create_all(bind=engine)

    get_embedder()  # ~1 s ONNX load, done once here instead of mid-request

    retriever = get_retriever()
    count = retriever.store.count()
    if count == 0:
        log.warning("kb.empty", hint="run: uv run python -m app.rag.ingest --reset")
    else:
        retriever._ensure_bm25()
        log.info("kb.ready", chunks=count)

    llm = get_llm()
    if await llm.health():
        log.info("llm.ready")
    else:
        log.warning("llm.unreachable", base_url=settings.llm_base_url)

    yield

    await llm.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Valorant Voice Coach",
        version="0.1.0",
        description="Personalized, RAG-grounded Valorant coaching over a local LLM.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # local single-user MVP
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(web.router)
    app.include_router(admin.router)
    app.include_router(chat.router)
    app.include_router(profile.router)
    return app


app = create_app()
