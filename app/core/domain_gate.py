"""Keep the assistant strictly on Valorant.

Two reasons this exists as code rather than as a line in the system prompt:

  1. A 3B model does not reliably hold a refusal instruction under pressure.
  2. Rejecting off-topic input before the LLM call saves a full generation -
     the check costs well under a millisecond.

The gate is intentionally permissive about follow-ups ("what about B site?")
because conversational context makes short queries topic-less on their own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.core.vocab import MULTIWORD_TERMS, SINGLE_TERMS

# Short conversational continuations that carry no topic of their own. These
# are allowed through when the conversation is already about Valorant.
_FOLLOW_UP = re.compile(
    r"^(and|what about|how about|why|why not|explain|more|go on|continue|"
    r"repeat|again|ok|okay|thanks|yes|no|sure)\b",
    re.IGNORECASE,
)

# Things people try when steering the assistant off-task.
_OFF_TOPIC = {
    "recipe", "weather", "python", "javascript", "homework", "essay",
    "stock", "crypto", "bitcoin", "medical", "diagnosis", "lawyer",
    "legal advice", "poem", "song lyrics", "translate", "resume", "cv",
    "minecraft", "fortnite", "league of legends", "csgo", "counter-strike",
    "overwatch", "apex legends", "dota",
}

REFUSAL = (
    "Nah, that's not my lane, I only coach Valorant. "
    "Hit me with maps, agents, util, site executes, rotates, "
    "economy, team comps, or whatever's keeping you hardstuck."
)


@dataclass
class GateResult:
    allowed: bool
    score: float
    reason: str = ""


def _tokens(text: str) -> list[str]:
    # Split on the apostrophe too, so "Jett's" yields "jett" and matches the
    # vocabulary. Keeping it would produce the token "jett's", which never does.
    return re.findall(r"[a-z0-9]+", text.lower())


def check(query: str, has_history: bool = False) -> GateResult:
    if not settings.domain_gate_enabled:
        return GateResult(True, 1.0, "disabled")

    lowered = query.lower()
    tokens = _tokens(query)
    if not tokens:
        return GateResult(False, 0.0, "empty")

    # Explicit competitor / off-domain mentions are refused outright, even if
    # the query also contains Valorant words ("is Valorant better than CSGO").
    for term in _OFF_TOPIC:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return GateResult(False, 0.0, f"off_topic:{term}")

    hits = sum(1 for t in tokens if t in SINGLE_TERMS)
    # Phrases ("force buy", "post plant", "team comp") cannot match a single
    # token, so they are matched against the raw text instead.
    hits += sum(1 for phrase in MULTIWORD_TERMS if phrase in lowered)
    score = hits / max(len(tokens), 1)

    # Recall matters far more than precision here. Wrongly refusing a real
    # coaching question ("how do I stop losing rounds after we plant?") is a
    # much worse experience than answering a borderline one, and the system
    # prompt is a second line of defence. So a single domain term is enough;
    # the density score is kept only for logging and tuning.
    if hits >= 1:
        return GateResult(True, score, "domain_term")
    if has_history and len(tokens) <= 8 and _FOLLOW_UP.match(query.strip()):
        return GateResult(True, score, "follow_up")

    return GateResult(False, score, "no_domain_signal")
