"""Answer-quality bake-off between two local models.

Speed is measured by bench_llm.py; this measures whether the bigger model is
actually worth its VRAM. Same profile, same retrieved notes, same prompt --
only the model changes, so differences are attributable.

    uv run python scripts/compare_models.py
    uv run python scripts/compare_models.py --a llama3.2:3b-instruct-q4_K_M \
                                            --b llama3.1:8b-instruct-q4_K_M

Context length is a load-time property of the model, so to compare different
context sizes create Modelfile variants first (see scripts/setup_ollama.ps1)
and pass those names here.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time

from app.core.filters import build_context
from app.core.prompt import build_messages
from app.db.models import Base
from app.db.repositories import ProfileRepo
from app.db.session import SessionLocal, engine
from app.llm.ollama_client import OllamaClient
from app.logging_setup import configure_logging
from app.rag.chroma_store import ChromaStore
from app.rag.retriever import Retriever
from app.schemas.profile import ProfileIn

USERNAME = "bakeoff"

PROFILE = ProfileIn(
    current_rank="gold",
    target_rank="platinum",
    preferred_role="duelist",
    playstyle="aggressive",
    main_agents=["jett"],
    favorite_maps=["ascent"],
    strengths=["aim"],
    weaknesses=["overpeeking", "economy"],
)

QUESTIONS = [
    "Give me an attack strategy for Ascent",
    "How do I stop losing rounds after we plant?",
    "When should we force buy?",
    "What is holding me back in gold?",
]


def score_format(reply: str) -> str:
    """Cheap, objective checks on whether the model followed the format rules."""
    words = len(reply.split())
    bullets = len(re.findall(r"^\s*-\s+", reply, flags=re.MULTILINE))
    has_watch = "watch out" in reply.lower()
    markdown = bool(re.search(r"[*#`]|\-\s+\-", reply))

    flags = []
    flags.append("len OK" if words <= 150 else f"len {words}w OVER")
    flags.append("bullets OK" if 3 <= bullets <= 4 else f"bullets {bullets} BAD")
    flags.append("watch-out OK" if has_watch else "watch-out MISSING")
    flags.append("clean" if not markdown else "MARKDOWN LEAK")
    return " | ".join(flags)


async def run_model(model: str, retriever: Retriever, session) -> None:
    print("\n" + "=" * 74)
    print(f"MODEL: {model}")
    print("=" * 74)

    client = OllamaClient(model=model)
    profile = ProfileRepo(session).get(USERNAME)

    # Warm-up: the first call pays the model load, which is not what we measure.
    await client.complete([{"role": "user", "content": "hi"}])

    try:
        for q in QUESTIONS:
            ctx = build_context(q, profile)
            chunks = retriever.search(ctx)
            messages = build_messages(q, ctx, chunks, profile, [])

            t0 = time.perf_counter()
            reply = (await client.complete(messages)).strip()
            elapsed = time.perf_counter() - t0

            print(f"\nQ: {q}")
            print(f"{reply}\n")
            print(f"   [{elapsed:.2f}s] {score_format(reply)}")
    finally:
        await client.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="llama3.2:3b-instruct-q4_K_M")
    parser.add_argument("--b", default="llama3.1:8b-instruct-q4_K_M")
    args = parser.parse_args()

    configure_logging()
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    ProfileRepo(session).upsert(USERNAME, PROFILE)
    retriever = Retriever(ChromaStore())

    print(f"profile: {PROFILE.current_rank} | {PROFILE.main_agents} | "
          f"{PROFILE.playstyle} | goal {PROFILE.target_rank}")

    await run_model(args.a, retriever, session)
    await run_model(args.b, retriever, session)

    session.close()
    print("\nRun `ollama ps` during each pass to confirm 100% GPU.")


if __name__ == "__main__":
    asyncio.run(main())
