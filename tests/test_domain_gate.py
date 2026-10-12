"""The gate must let real coaching questions through and keep everything else out.

Recall is the priority: a wrongly refused question is a visibly broken
assistant, while a borderline one still gets caught by the system prompt.
"""

from __future__ import annotations

import pytest

from app.core import domain_gate

ON_TOPIC = [
    "Give me an attack strategy for Ascent",
    "How do I stop losing rounds after we plant?",
    "What should I buy on a force round?",
    "Which agent should I main to climb out of gold?",
    "My crosshair placement is bad, how do I fix it?",
    "Best team comp for Split?",
    "How do I retake B site?",
    "why do we keep losing rounds",
    "What does Jett's Tailwind do?",
    "how should i rotate on haven",
    # Phrasings whose only domain signal is a multi-word term. These were all
    # wrongly refused before the gate learned to match phrases.
    "When should we force buy?",
    "Is a full buy worth it here?",
    "What do we do on a bonus round?",
    "how do i improve my post plant",
    "what team comp should we run",
    "should we save or force",
    # Questions about the player's own profile.
    "check my profile give me the tip",
    "what are my weaknesses",
    "give me tips",
]

OFF_TOPIC = [
    "What is the best pasta recipe?",
    "Write me a Python script to sort a list",
    "What is the weather tomorrow?",
    "Help me write my resume",
    "Should I buy bitcoin?",
    "Is Valorant better than CSGO?",
]


@pytest.mark.parametrize("query", ON_TOPIC)
def test_allows_coaching_questions(query: str) -> None:
    result = domain_gate.check(query)
    assert result.allowed, f"wrongly refused: {query!r} ({result.reason})"


@pytest.mark.parametrize("query", OFF_TOPIC)
def test_refuses_off_topic(query: str) -> None:
    result = domain_gate.check(query)
    assert not result.allowed, f"wrongly allowed: {query!r} ({result.reason})"


def test_follow_up_needs_history() -> None:
    assert not domain_gate.check("what about B?", has_history=False).allowed
    assert domain_gate.check("what about B?", has_history=True).allowed


def test_empty_query_refused() -> None:
    assert not domain_gate.check("   ").allowed


def test_no_vocabulary_term_is_unreachable() -> None:
    """Every term must be matchable by the strategy it is routed to.

    A multi-word entry in the single-token set (or vice versa) is dead weight
    that silently never fires, which is exactly how "force buy" broke.
    """
    from app.core.vocab import MULTIWORD_TERMS, SINGLE_TERMS

    assert all(" " not in t and "-" not in t for t in SINGLE_TERMS)
    assert all(" " in t or "-" in t for t in MULTIWORD_TERMS)

    for phrase in MULTIWORD_TERMS:
        assert domain_gate.check(f"tell me about {phrase}").allowed, phrase
