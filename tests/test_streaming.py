"""Sentence splitting drives the voice latency budget, so it gets real tests.

The tricky case is Valorant text specifically: "A." and "B." appear constantly
as site names, and splitting on them would send fragments to the synthesizer.
"""

from __future__ import annotations

from app.speech.streaming import clean_for_speech, sentences_from_text, split_sentences


def test_splits_on_sentence_end() -> None:
    text = "Smoke Heaven first. Then flash A Main and enter together."
    assert list(sentences_from_text(text)) == [
        "Smoke Heaven first.",
        "Then flash A Main and enter together.",
    ]


def test_short_first_fragment_is_held_until_long_enough() -> None:
    # "Sure." alone is below FIRST_MIN_CHARS, so it must merge forward rather
    # than being spoken as a one-word utterance.
    sentences, rest = split_sentences("Sure. ", is_first=True)
    assert sentences == []
    assert rest == "Sure. "


def test_incomplete_tail_is_returned_as_remainder() -> None:
    sentences, rest = split_sentences(
        "Smoke Heaven and flash A Main first. Then you sh", is_first=True
    )
    assert sentences == ["Smoke Heaven and flash A Main first."]
    assert rest.strip() == "Then you sh"


def test_newline_is_a_boundary() -> None:
    text = "- Smoke Heaven before the entry\n- Flash over A Main and go\n"
    assert list(sentences_from_text(text)) == [
        "- Smoke Heaven before the entry",
        "- Flash over A Main and go",
    ]


def test_clean_for_speech_strips_markdown() -> None:
    assert clean_for_speech("- **Smoke** Heaven `first`") == "Smoke Heaven first"


def test_nothing_is_lost() -> None:
    text = "Take Mid control first. Smoke Heaven and Hell. Then enter A Main."
    assert "".join(sentences_from_text(text)).replace(" ", "") == text.replace(" ", "")
