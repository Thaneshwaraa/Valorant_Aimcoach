"""Text client. Talks to the running API over HTTP.

    uv run python clients/cli_chat.py --setup     # create/update your profile
    uv run python clients/cli_chat.py             # chat

Use this to iterate on prompts and the knowledge base; it exercises exactly
the same server path the voice client does, minus the audio.
"""

from __future__ import annotations

import argparse
import sys
import uuid

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

RANKS = ["iron", "bronze", "silver", "gold", "platinum", "diamond",
         "ascendant", "immortal", "radiant"]
ROLES = ["duelist", "controller", "initiator", "sentinel", "flex"]
PLAYSTYLES = ["aggressive", "passive", "balanced"]


def _csv(prompt: str, default: str = "") -> list[str]:
    raw = Prompt.ask(prompt, default=default)
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def setup_profile(base: str, username: str) -> None:
    console.print(Panel(f"Profile setup for [bold]{username}[/bold]", style="cyan"))

    payload = {
        "current_rank": Prompt.ask("Current rank", choices=RANKS, default="gold"),
        "target_rank": Prompt.ask("Target rank", choices=RANKS, default="platinum"),
        "preferred_role": Prompt.ask("Preferred role", choices=ROLES, default="duelist"),
        "playstyle": Prompt.ask("Playstyle", choices=PLAYSTYLES, default="balanced"),
        "main_agents": _csv("Main agents (comma separated)", "jett"),
        "favorite_maps": _csv("Favourite maps (comma separated)", "ascent"),
        "strengths": _csv("Strengths (comma separated)", "aim"),
        "weaknesses": _csv("Weaknesses (comma separated)", "overpeeking"),
        "notes": Prompt.ask("Anything else", default=""),
    }

    r = httpx.put(f"{base}/profile/{username}", json=payload, timeout=30)
    r.raise_for_status()
    console.print("[green]Profile saved.[/green]\n")


def chat_loop(base: str, username: str, use_rag: bool, show_sources: bool) -> None:
    session_id = uuid.uuid4().hex
    console.print(
        Panel(
            "Valorant Voice Coach - text mode\n"
            "Ask about maps, agents, strategy, economy or ranking up.\n"
            "Type 'exit' to quit.",
            style="cyan",
        )
    )

    with httpx.Client(timeout=180) as client:
        while True:
            try:
                question = Prompt.ask("\n[bold cyan]you[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                break
            if question.strip().lower() in {"exit", "quit", "q"}:
                break
            if not question.strip():
                continue

            payload = {
                "username": username,
                "message": question,
                "session_id": session_id,
                "use_rag": use_rag,
            }

            console.print("\n[bold magenta]coach[/bold magenta]")
            reply_parts: list[str] = []
            try:
                with client.stream(
                    "POST", f"{base}/chat/stream", json=payload
                ) as resp:
                    resp.raise_for_status()
                    for text in resp.iter_text():
                        if text:
                            reply_parts.append(text)
                            console.print(text, end="", highlight=False)
            except httpx.HTTPError as exc:
                console.print(f"[red]request failed: {exc}[/red]")
                continue
            console.print()

            if show_sources:
                # The streaming endpoint returns text only, so ask the
                # buffered endpoint for the sources behind that answer.
                try:
                    meta = client.post(
                        f"{base}/chat", json={**payload, "message": question}
                    ).json()
                    for s in meta.get("sources", []):
                        console.print(f"  [dim]- {s['title']} ({s['score']})[/dim]")
                except httpx.HTTPError:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--user", default="default")
    parser.add_argument("--setup", action="store_true", help="edit your profile")
    parser.add_argument("--no-rag", action="store_true", help="disable retrieval")
    parser.add_argument("--sources", action="store_true", help="show retrieved notes")
    args = parser.parse_args()

    try:
        httpx.get(f"{args.base}/health", timeout=5).raise_for_status()
    except httpx.HTTPError:
        console.print(
            f"[red]Cannot reach the API at {args.base}.[/red]\n"
            "Start it with: uv run uvicorn app.main:app --reload"
        )
        return 1

    if args.setup:
        setup_profile(args.base, args.user)

    chat_loop(args.base, args.user, use_rag=not args.no_rag, show_sources=args.sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
