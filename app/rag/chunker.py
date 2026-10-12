"""Markdown + YAML frontmatter -> retrievable chunks.

Splitting happens on "## " headings because the knowledge base is authored
that way: each heading is already one self-contained coaching idea. Oversized
sections are split again on paragraph boundaries so no single chunk blows the
prompt budget.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import frontmatter

from app.logging_setup import get_logger
from app.rag.store import Chunk

log = get_logger(__name__)

MAX_WORDS = 220  # ~290 tokens; six of these fit comfortably in a 4096 ctx
MIN_WORDS = 12  # drop stubs that would only add noise to the index

_HEADING = re.compile(r"^##\s+(.+)$", re.MULTILINE)

# Section headings name the site far more often than frontmatter does
# ("## A site execute", "## B site hold"). Without this, every chunk in a map
# document carries the same empty site tag, and a question about B site
# happily retrieves the A site section.
_SITE_IN_HEADING = re.compile(r"\b([abc])\s*(?:site|main|long|short)\b", re.IGNORECASE)


def _infer_site(heading: str) -> str:
    """Best-effort site tag from a heading. Empty means "applies to all"."""
    m = _SITE_IN_HEADING.search(heading)
    return m.group(1).upper() if m else ""


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\|]", value)
    else:
        parts = list(value)
    return [str(p).strip().lower() for p in parts if str(p).strip()]


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Return (heading, text) pairs. Text before the first heading is kept."""
    matches = list(_HEADING.finditer(body))
    if not matches:
        return [("", body.strip())]

    sections: list[tuple[str, str]] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((m.group(1).strip(), body[m.end() : end].strip()))
    return sections


def _pack_paragraphs(text: str) -> list[str]:
    """Greedily merge paragraphs up to MAX_WORDS."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    packed: list[str] = []
    buf: list[str] = []
    count = 0
    for p in paragraphs:
        n = len(p.split())
        if buf and count + n > MAX_WORDS:
            packed.append("\n\n".join(buf))
            buf, count = [p], n
        else:
            buf.append(p)
            count += n
    if buf:
        packed.append("\n\n".join(buf))
    return packed


def chunk_file(path: Path, kb_root: Path) -> list[Chunk]:
    post = frontmatter.load(path)
    meta = post.metadata
    rel = path.relative_to(kb_root).as_posix()
    doc_id = str(meta.get("id") or path.stem).strip().lower()
    doc_title = str(meta.get("title") or path.stem.replace("-", " ").title())

    doc_site = str(meta.get("site", "")).strip().upper()

    common = {
        "source": rel,
        "map": str(meta.get("map", "")).strip().lower(),
        "side": str(meta.get("side", "")).strip().lower(),
        "topic": _as_list(meta.get("topic")),
        "agents": _as_list(meta.get("agents")),
        "roles": _as_list(meta.get("roles")),
        "rank_band": _as_list(meta.get("rank_band")),
        "playstyle": _as_list(meta.get("playstyle")),
    }

    chunks: list[Chunk] = []
    for heading, section in _split_sections(post.content):
        # Frontmatter wins when present; otherwise fall back to the heading.
        site = doc_site or _infer_site(heading)
        for piece in _pack_paragraphs(section):
            if len(piece.split()) < MIN_WORDS:
                continue
            title = f"{doc_title} - {heading}" if heading else doc_title
            # Prepending the title gives both BM25 and the embedder the map and
            # agent names even when the body text only says "this site".
            text = f"{title}\n\n{piece}"
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{digest}", text=text, title=title, site=site, **common
                )
            )
    return chunks


def chunk_directory(kb_dir: Path) -> list[Chunk]:
    files = sorted(kb_dir.rglob("*.md"))
    all_chunks: list[Chunk] = []
    seen: set[str] = set()
    for f in files:
        try:
            for c in chunk_file(f, kb_dir):
                if c.id in seen:  # identical text in two docs
                    continue
                seen.add(c.id)
                all_chunks.append(c)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop ingest
            log.error("chunk.failed", file=str(f), error=str(exc))
    log.info("chunk.done", files=len(files), chunks=len(all_chunks))
    return all_chunks
