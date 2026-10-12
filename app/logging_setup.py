"""Structured logging.

The voice pipeline has a latency budget (see README). Timing every stage as
structured key/value pairs is what makes that budget debuggable; plain print()
output cannot be filtered or aggregated.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

import structlog

from app.config import settings


def configure_logging() -> None:
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=settings.log_level.upper()
    )
    # These libraries log every HTTP call at INFO, which drowns our own timing
    # lines during ingest and streaming.
    for noisy in ("httpx", "httpcore", "chromadb", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)


@contextmanager
def timed(logger, stage: str, **kw):
    """Log how long a pipeline stage took, in milliseconds."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info(
            "stage", stage=stage, ms=round((time.perf_counter() - t0) * 1000, 1), **kw
        )
