"""Knowledge-base ingestion.

Run after editing anything under data/kb:

    uv run python -m app.rag.ingest --reset

Always a full rebuild rather than an incremental update. The corpus is small
(seconds to rebuild) and full rebuilds make deleted or renamed documents
disappear correctly, which incremental upserts silently get wrong.
"""

from __future__ import annotations

import argparse
import sys

from app.config import settings
from app.logging_setup import configure_logging, get_logger
from app.rag.chroma_store import ChromaStore
from app.rag.chunker import chunk_directory
from app.rag.embedder import get_embedder

log = get_logger(__name__)

BATCH = 64


def ingest(reset: bool = True) -> int:
    kb_dir = settings.abs_path(settings.kb_dir)
    if not kb_dir.exists():
        log.error("ingest.no_kb_dir", path=str(kb_dir))
        return 0

    chunks = chunk_directory(kb_dir)
    if not chunks:
        log.error("ingest.no_chunks", path=str(kb_dir))
        return 0

    store = ChromaStore()
    if reset:
        store.reset()

    embedder = get_embedder()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        vectors = embedder.embed_documents([c.text for c in batch])
        store.add(batch, vectors)
        log.info("ingest.batch", done=min(i + BATCH, len(chunks)), total=len(chunks))

    total = store.count()
    log.info("ingest.done", chunks=total, collection=settings.chroma_collection)
    return total


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest the Valorant knowledge base")
    parser.add_argument(
        "--reset", action="store_true", default=True, help="rebuild from scratch"
    )
    parser.add_argument("--no-reset", dest="reset", action="store_false")
    args = parser.parse_args()
    return 0 if ingest(reset=args.reset) else 1


if __name__ == "__main__":
    sys.exit(main())
