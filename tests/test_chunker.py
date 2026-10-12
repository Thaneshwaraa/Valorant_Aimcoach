"""Chunking and metadata round-tripping."""

from __future__ import annotations

from pathlib import Path

from app.rag.chunker import chunk_directory, chunk_file
from app.rag.store import Chunk

DOC = """---
id: test-doc
title: Test Doc
map: ascent
side: attack
site: A
topic: [execute, entry]
agents: [jett, sova]
roles: [duelist]
rank_band: [mid]
playstyle: [aggressive]
---

## First section

This is the first section and it contains enough words to survive the minimum
word filter that drops tiny fragments from the index entirely.

## Second section

This is the second section and it also contains enough words to survive the
minimum word filter that drops tiny fragments from the index entirely.
"""


def test_chunk_file_splits_on_headings(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(DOC, encoding="utf-8")

    chunks = chunk_file(path, tmp_path)
    assert len(chunks) == 2

    first = chunks[0]
    assert first.map == "ascent"
    assert first.side == "attack"
    assert first.site == "A"
    assert first.agents == ["jett", "sova"]
    assert first.rank_band == ["mid"]
    assert first.source == "doc.md"
    # The title is prepended so BM25 and the embedder both see "Ascent".
    assert first.text.startswith("Test Doc - First section")


def test_chunk_ids_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(DOC, encoding="utf-8")
    assert [c.id for c in chunk_file(path, tmp_path)] == [
        c.id for c in chunk_file(path, tmp_path)
    ]


def test_metadata_round_trip() -> None:
    original = Chunk(
        id="x:1",
        text="body",
        title="T",
        source="a/b.md",
        map="ascent",
        side="attack",
        site="A",
        topic=["execute"],
        agents=["jett"],
        roles=["duelist"],
        rank_band=["mid"],
        playstyle=["aggressive"],
    )
    restored = Chunk.from_metadata("x:1", "body", original.to_metadata())
    assert restored.agents == ["jett"]
    assert restored.rank_band == ["mid"]
    assert restored.map == "ascent"
    assert restored.source == "a/b.md"


SITE_DOC = """---
id: site-doc
title: Ascent Attack
map: ascent
side: attack
---

## A site execute

Smoke Heaven and Hell, plus a smoke or wall on CT, then flash over A Main and
enter together with a second player pinching from A Short.

## B site execute

Smoke Boathouse, Lane and CT, then flash into B Main and enter with two
players while a third comes from Market to catch the rotation.

## Mid control default

Send two players to Catwalk and one to contest Market, then smoke Mid Link so
the defenders cannot hold the angle from Tiles.
"""


def test_site_inferred_from_heading(tmp_path: Path) -> None:
    """The knowledge base tags sites in headings, not frontmatter.

    Without this inference every chunk in a map document shares one empty site
    tag, and a question about B site retrieves the A site section.
    """
    path = tmp_path / "sites.md"
    path.write_text(SITE_DOC, encoding="utf-8")

    by_title = {c.title.split(" - ")[-1]: c for c in chunk_file(path, tmp_path)}
    assert by_title["A site execute"].site == "A"
    assert by_title["B site execute"].site == "B"
    # Sections that apply to the whole map stay untagged, so they remain
    # eligible for any site question.
    assert by_title["Mid control default"].site == ""


def test_frontmatter_site_wins_over_heading(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(DOC, encoding="utf-8")  # frontmatter declares site: A
    assert all(c.site == "A" for c in chunk_file(path, tmp_path))


def test_real_kb_has_site_tagged_chunks() -> None:
    """Guards against the tagging silently reverting to all-empty."""
    kb = Path(__file__).resolve().parent.parent / "data" / "kb"
    sites = {c.site for c in chunk_directory(kb) if c.site}
    assert {"A", "B"} <= sites, f"expected A and B site tags, got {sites}"


def test_real_kb_ingests_cleanly() -> None:
    """The shipped knowledge base must parse without errors."""
    kb = Path(__file__).resolve().parent.parent / "data" / "kb"
    chunks = chunk_directory(kb)
    assert len(chunks) > 50
    assert len({c.id for c in chunks}) == len(chunks), "duplicate chunk ids"
    assert all(c.title for c in chunks)
