"""Turn a token stream into speakable sentences.

This is the single most important piece of the voice latency budget. Waiting
for the full LLM response before synthesising costs 4-6 seconds of silence;
emitting each sentence as it completes drops the perceived delay to about one
second, because the first sentence is spoken while the rest is still being
generated.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator

# A sentence ends at . ! ? or a newline, but not on a decimal or an
# abbreviation. Valorant text is full of "A." and "B." site references, so the
# lookahead requires whitespace plus a capital or end-of-string.
_BOUNDARY = re.compile(r"(?<=[.!?])(?=\s|$)|\n+")

_ABBREVIATIONS = {"mr.", "mrs.", "vs.", "e.g.", "i.e.", "no.", "approx."}

# Speak the first fragment early even if it is short, so audio starts sooner.
# Low enough that a real opening line ("Smoke Heaven first.") is spoken
# immediately, high enough that filler ("Sure.") merges into what follows.
FIRST_MIN_CHARS = 15
MIN_CHARS = 45


def _looks_complete(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    last = stripped.split()[-1].lower()
    return last not in _ABBREVIATIONS


def split_sentences(buffer: str, is_first: bool) -> tuple[list[str], str]:
    """Split off every complete sentence, returning them plus the remainder."""
    minimum = FIRST_MIN_CHARS if is_first else MIN_CHARS
    out: list[str] = []
    rest = buffer

    while True:
        match = _BOUNDARY.search(rest)
        if not match:
            break
        head, tail = rest[: match.end()], rest[match.end() :]
        if len(head.strip()) < minimum or not _looks_complete(head):
            # Too short or ended on an abbreviation: keep accumulating.
            # Most boundaries are zero-width lookaheads, so searching from
            # match.end() would rediscover the same position forever.
            resume = match.end() + (1 if match.end() == match.start() else 0)
            nxt = _BOUNDARY.search(rest, resume)
            if not nxt:
                break
            head, tail = rest[: nxt.end()], rest[nxt.end() :]
        cleaned = head.strip()
        if cleaned:
            out.append(cleaned)
            minimum = MIN_CHARS
        rest = tail

    return out, rest


def sentences_from_text(text: str) -> Iterator[str]:
    """Non-streaming variant, used for tests and for buffered replies."""
    sentences, rest = split_sentences(text, is_first=True)
    yield from sentences
    if rest.strip():
        yield rest.strip()


async def sentences_from_stream(
    stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Yield speakable sentences from a stream of token deltas."""
    buffer = ""
    first = True
    async for delta in stream:
        buffer += delta
        sentences, buffer = split_sentences(buffer, is_first=first)
        for s in sentences:
            first = False
            yield s
    if buffer.strip():
        yield buffer.strip()


def clean_for_speech(text: str) -> str:
    """Strip the markdown the model emits despite being told not to.

    Piper reads punctuation literally, so a stray asterisk becomes an audible
    artefact. Cheaper to clean here than to keep fighting the prompt.
    """
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
