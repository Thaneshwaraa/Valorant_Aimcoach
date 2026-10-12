"""Prompt construction.

This file is where most of the answer quality actually comes from, so it is
written to be edited often. Three principles it follows:

  1. Ground hard. The model is told to use the retrieved context and to say
     when it does not know. A 3B model invents plausible-sounding callouts
     otherwise, and a wrong callout is worse than no answer.
  2. Personalise explicitly. The profile is rendered as a compact block and the
     model is instructed to use it, rather than being left to infer relevance.
  3. Stay short. Answers are spoken aloud - long paragraphs are unusable in
     voice. The length rule is stated in tokens the model can feel.
"""

from __future__ import annotations

from app.core.filters import QueryContext
from app.core.vocab import RANK_BANDS
from app.llm.base import ChatMessage
from app.rag.store import Chunk
from app.schemas.profile import ProfileOut

SYSTEM = """You are a Valorant coach speaking to one player. You discuss only \
Valorant.

Use the COACHING NOTES as your only source of facts. Never state a callout, \
ability effect, lineup, or timing that is not in the notes. If the notes do \
not cover it, say so in one sentence and give a general principle instead. \
Do not repeat the same advice twice, and do not describe a second site or a \
second plan the player did not ask about. Write your own sentences rather \
than copying a note heading.

Write for the PLAYER PROFILE: their rank, main agent, role and playstyle. \
Address them as "you". Make it personal: line 1 names their main agent or \
rank, and if WEAKNESSES are listed, at least one bullet fixes one of them by \
name. Never ask the player for their profile; you already have it below.

Always answer in this shape, and always include every part:

Line 1: one sentence naming the plan.
Then exactly 3 or 4 bullet lines. Begin each bullet line with a single hyphen \
followed by one space, and never more than one hyphen. Each bullet is one \
short sentence giving a concrete action drawn from the notes.
Last line: one sentence starting with "Watch out:" naming the single mistake \
to avoid at their rank.

Never write fewer than 3 bullets and never more than 4. Stay under 150 words. \
Plain text only, no headers, no bold, no emoji. If the question is not about \
Valorant, reply with one sentence declining.

VOICE: sound like a ranked teammate on comms, not a textbook. Write in your \
own words with contractions and everyday Valorant lingo such as util, swing, \
dry peek, trade, off-angle, rotate, retake, post-plant, eco, force, creds. \
Keep it chill and clear: no "fam", no memes, no all-caps."""

_BAND_GUIDANCE = {
    "low": (
        "This player is in the lower ranks. Prioritise fundamentals: crosshair "
        "placement, not over-peeking, using utility before entering, and simple "
        "default setups. Avoid complex team-dependent executes."
    ),
    "mid": (
        "This player is in the middle ranks. They can execute basic utility "
        "combos. Focus on map control, trade discipline, economy decisions, and "
        "punishing common habits at this level."
    ),
    "high": (
        "This player is in the high ranks. Assume strong mechanics and game "
        "sense. Focus on timing, mid-round adaptation, utility efficiency, and "
        "reading opponent tendencies."
    ),
}

_PLAYSTYLE_GUIDANCE = {
    "aggressive": (
        "They play aggressively: give them entry angles, fast-execute timings, "
        "and aggressive utility usage, but include the discipline cue that "
        "stops the aggression from becoming a free death."
    ),
    "passive": (
        "They play passively: give them supportive positioning, safe trades, "
        "post-plant angles, and ways to take space without dry-peeking."
    ),
    "balanced": (
        "They play a balanced style: give them a clear read on when to take "
        "space and when to play for information."
    ),
}


def render_profile(profile: ProfileOut | None) -> str:
    if profile is None:
        return "PLAYER PROFILE: unknown. Give generally applicable advice."

    if not profile.current_rank or profile.current_rank == "unranked":
        # Without this the model picks a rank at random and coaches for it.
        rank_line = (
            "- Current rank: NOT SET. Do not name or guess a rank anywhere in "
            "your answer; give advice that holds at any rank."
        )
    else:
        rank_line = f"- Current rank: {profile.current_rank}"

    lines = [rank_line]
    if profile.target_rank:
        lines.append(f"- Goal rank: {profile.target_rank}")
    if profile.main_agents:
        lines.append(f"- Main agents: {', '.join(profile.main_agents)}")
    if profile.preferred_role:
        lines.append(f"- Preferred role: {profile.preferred_role}")
    lines.append(f"- Playstyle: {profile.playstyle}")
    if profile.favorite_maps:
        lines.append(f"- Favourite maps: {', '.join(profile.favorite_maps)}")
    if profile.strengths:
        lines.append(f"- Strengths: {', '.join(profile.strengths)}")
    if profile.weaknesses:
        lines.append(f"- Weaknesses: {', '.join(profile.weaknesses)}")
    if profile.notes:
        lines.append(f"- Notes: {profile.notes}")
    if profile.edpi:
        lines.append(
            f"- Mouse: {profile.dpi} DPI, {profile.sensitivity:g} sensitivity "
            f"(eDPI {profile.edpi})"
        )

    band = RANK_BANDS.get(profile.current_rank, "")
    guidance = [_BAND_GUIDANCE.get(band, ""), _PLAYSTYLE_GUIDANCE.get(profile.playstyle, "")]
    tail = " ".join(g for g in guidance if g)

    block = "PLAYER PROFILE:\n" + "\n".join(lines)
    return f"{block}\n\nCOACHING APPROACH: {tail}" if tail else block


def render_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return (
            "COACHING NOTES: none matched this question. Answer from general "
            "Valorant principles and say clearly that you lack specific notes."
        )

    parts = []
    for i, c in enumerate(chunks, 1):
        # The chunker prepends the document title to the body so BM25 and the
        # embedder both see the map and agent names. Strip it back off here:
        # a human-readable heading sitting in the context gets copied verbatim
        # into the answer, however firmly the prompt says not to. What the
        # model actually needs is the scope, so emit that as a compact tag
        # instead -- it carries the same information in a shape nobody would
        # mistake for a title to repeat.
        title, _, body = c.text.partition("\n\n")
        body = (body.strip() or title).strip()

        tags = [t for t in (c.map, c.side, f"{c.site} site" if c.site else "") if t]
        scope = f" ({', '.join(tags)})" if tags else ""
        parts.append(f"[{i}]{scope}\n{body}")

    return "COACHING NOTES:\n\n" + "\n\n".join(parts)


def build_messages(
    query: str,
    ctx: QueryContext,
    chunks: list[Chunk],
    profile: ProfileOut | None,
    history: list[tuple[str, str]] | None = None,
) -> list[ChatMessage]:
    """Assemble the final message list.

    Ordering matters: the profile and notes go in the system turn so they are
    not mistaken for something the user said, and history sits between them and
    the live question so the model reads it as prior conversation.
    """
    system = "\n\n".join([SYSTEM, render_profile(profile), render_context(chunks)])
    messages: list[ChatMessage] = [{"role": "system", "content": system}]

    for role, content in (history or []):
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    hint = ""
    if ctx.maps:
        hint = f" (map: {ctx.maps[0]}{', ' + ctx.side if ctx.side else ''})"
    messages.append({"role": "user", "content": f"{query}{hint}"})
    return messages
