"""Measure the numbers the voice latency budget depends on.

Reports time-to-first-token and tokens/second against a realistic prompt size,
after a warm-up pass so the model load is not counted.

    uv run python scripts/bench_llm.py
    uv run python scripts/bench_llm.py --model llama3.1:8b-instruct-q4_K_M

Targets for a usable voice loop: TTFT under 0.4 s, throughput above 40 tok/s.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from app.config import settings
from app.llm.base import ChatMessage
from app.llm.ollama_client import OllamaClient

# Roughly the size of a real request: system prompt + profile + six retrieved
# chunks. Benchmarking on a two-word prompt would flatter the prefill stage.
FILLER = (
    "Ascent is a two-site map built around a large open middle. Mid is the "
    "pivot: the team that owns Mid can hit either site quickly and can cut "
    "rotations. Heaven overlooks A site from the defender side. "
)

PROMPT = "Give me a short attack plan for Ascent A site."


def _messages(ctx_words: int) -> list[ChatMessage]:
    repeats = max(1, ctx_words // len(FILLER.split()))
    return [
        {"role": "system", "content": "You are a Valorant coach.\n\n" + FILLER * repeats},
        {"role": "user", "content": PROMPT},
    ]


async def run_once(client: OllamaClient, messages: list[ChatMessage]) -> tuple[float, int, float]:
    t0 = time.perf_counter()
    ttft = -1.0
    chunks = 0
    async for _delta in client.stream(messages):
        if ttft < 0:
            ttft = time.perf_counter() - t0
        chunks += 1
    total = time.perf_counter() - t0
    return ttft, chunks, total


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=settings.llm_model)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ctx-words", type=int, default=1000)
    args = parser.parse_args()

    client = OllamaClient(model=args.model)
    messages = _messages(args.ctx_words)

    print(f"model      : {args.model}")
    print(f"num_ctx    : {settings.llm_num_ctx}")
    print(f"max_tokens : {settings.llm_max_tokens}")
    print(f"prompt     : ~{args.ctx_words} words of context\n")

    print("warming up (loads the model into VRAM)...")
    await run_once(client, messages)

    ttfts, rates = [], []
    for i in range(args.runs):
        ttft, chunks, total = await run_once(client, messages)
        gen_time = max(total - ttft, 1e-6)
        rate = (chunks - 1) / gen_time
        ttfts.append(ttft)
        rates.append(rate)
        print(
            f"run {i + 1}: TTFT {ttft * 1000:7.1f} ms | "
            f"{chunks:4d} chunks | {rate:5.1f} tok/s | total {total:5.2f} s"
        )

    avg_ttft = sum(ttfts) / len(ttfts)
    avg_rate = sum(rates) / len(rates)
    print(f"\naverage: TTFT {avg_ttft * 1000:.1f} ms | {avg_rate:.1f} tok/s")

    ok_ttft = avg_ttft < 0.4
    ok_rate = avg_rate > 40
    print(f"TTFT  target <400ms : {'PASS' if ok_ttft else 'FAIL'}")
    print(f"Speed target >40tps : {'PASS' if ok_rate else 'FAIL'}")
    if not (ok_ttft and ok_rate):
        print("\nIf this fails, check `ollama ps` shows 100% GPU. If it shows")
        print("any CPU share, lower LLM_NUM_CTX or use a smaller quantization.")

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
