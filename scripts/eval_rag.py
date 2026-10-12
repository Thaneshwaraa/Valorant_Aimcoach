"""Measure retrieval quality against the golden question set.

The RAG layer is the part of this system most likely to silently get worse
when you tune something. This turns that into a number.

    uv run python scripts/eval_rag.py
    uv run python scripts/eval_rag.py --verbose      # show every miss

Reports recall@1 and recall@3: how often the document that should answer the
question appears in the top 1 and top 3 retrieved chunks. Treat recall@3 of
80% or better as the bar before spending effort on prompt tuning, because no
prompt can fix a note that was never retrieved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from app.core.filters import build_context
from app.logging_setup import configure_logging
from app.rag.chroma_store import ChromaStore
from app.rag.retriever import Retriever
from app.schemas.profile import ProfileOut

QUESTIONS = Path(__file__).resolve().parent.parent / "tests" / "eval" / "questions.yaml"


def _profile(data: dict | None) -> ProfileOut | None:
    if not data:
        return None
    return ProfileOut(
        user_id="eval",
        username="eval",
        current_rank=data.get("current_rank", "unranked"),
        target_rank=data.get("target_rank", ""),
        preferred_role=data.get("preferred_role", ""),
        playstyle=data.get("playstyle", "balanced"),
        main_agents=data.get("main_agents", []),
        favorite_maps=data.get("favorite_maps", []),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    configure_logging()

    cases = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    store = ChromaStore()
    if store.count() == 0:
        print("Vector store is empty. Run: uv run python -m app.rag.ingest --reset")
        return 1

    retriever = Retriever(store)

    hits_at_1 = 0
    hits_at_3 = 0
    misses: list[tuple[str, list[str], list[str]]] = []
    # Document-level recall cannot see a B-site question retrieving the A-site
    # section, because both live in the same file. This counts that directly.
    contradictions: list[tuple[str, str]] = []

    for case in cases:
        question = case["q"]
        expected = case["expect"]
        expected = [expected] if isinstance(expected, str) else expected

        ctx = build_context(question, _profile(case.get("profile")))
        results = retriever.search(ctx, top_k=args.top_k)
        sources = [c.source for c in results]

        for c in results:
            if ctx.site and c.site and c.site != ctx.site:
                contradictions.append((question, f"site {c.site}: {c.title}"))
            if ctx.side and c.side and c.side != ctx.side:
                contradictions.append((question, f"side {c.side}: {c.title}"))
            if ctx.maps and c.map and c.map not in ctx.maps:
                contradictions.append((question, f"map {c.map}: {c.title}"))

        if sources[:1] and sources[0] in expected:
            hits_at_1 += 1
        if any(s in expected for s in sources[:3]):
            hits_at_3 += 1
        else:
            misses.append((question, expected, sources[:3]))

    total = len(cases)
    print(f"\n{'=' * 68}")
    print(f"questions   : {total}")
    print(f"recall@1    : {hits_at_1}/{total}  ({hits_at_1 / total:.0%})")
    print(f"recall@3    : {hits_at_3}/{total}  ({hits_at_3 / total:.0%})")
    print(f"contradict. : {len(contradictions)}  (retrieved notes for the wrong map/side/site)")
    print(f"{'=' * 68}")

    if contradictions:
        print("\ncontradictory retrievals:")
        for question, detail in contradictions[:15]:
            print(f"  {question}\n     -> {detail}")

    if misses:
        print(f"\n{len(misses)} miss(es) at top-3:")
        for question, expected, got in misses:
            print(f"\n  Q: {question}")
            print(f"     expected: {expected}")
            print(f"     got     : {got}")

    if args.verbose and not misses:
        print("\nno misses")

    return 0 if hits_at_3 / total >= 0.8 and not contradictions else 2


if __name__ == "__main__":
    sys.exit(main())
