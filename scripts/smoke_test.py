"""End-to-end smoke test with no server running.

Exercises the full pipeline in-process: profile -> query context -> hybrid
retrieval -> prompt -> Ollama. Run this after changing anything in app/core or
app/rag, before bothering with the API.

    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import time

from app.api.deps import get_coach
from app.db.models import Base
from app.db.repositories import ProfileRepo
from app.db.session import SessionLocal, engine
from app.logging_setup import configure_logging
from app.schemas.profile import ProfileIn

USERNAME = "smoketest"

PROFILE = ProfileIn(
    current_rank="gold",
    target_rank="platinum",
    preferred_role="duelist",
    playstyle="aggressive",
    main_agents=["jett"],
    favorite_maps=["ascent"],
    strengths=["aim", "entry"],
    weaknesses=["overpeeking", "economy"],
)

QUESTIONS = [
    "Give me an attack strategy for Ascent",
    "How do I stop losing rounds after we plant?",
    "What is the best pasta recipe?",  # must be refused
]


async def main() -> None:
    configure_logging()
    Base.metadata.create_all(bind=engine)

    coach = get_coach()
    session = SessionLocal()
    ProfileRepo(session).upsert(USERNAME, PROFILE)

    print(f"\nProfile: {PROFILE.current_rank} | {PROFILE.main_agents} | "
          f"{PROFILE.playstyle} | goal {PROFILE.target_rank}\n")

    for q in QUESTIONS:
        print("=" * 72)
        print(f"Q: {q}\n")
        t0 = time.perf_counter()
        resp = await coach.answer(session, USERNAME, q)
        elapsed = time.perf_counter() - t0

        print(resp.reply)
        print()
        if resp.refused:
            print(">> REFUSED by domain gate")
        else:
            print(">> sources:")
            for s in resp.sources:
                print(f"   - {s.title}  (score {s.score})")
        print(f">> total {elapsed:.2f}s  timings={resp.timings_ms}\n")

    session.close()
    await coach.llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
