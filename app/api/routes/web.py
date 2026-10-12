"""Serves the browser UI at the site root.

Without this, http://127.0.0.1:8000/ returns a bare {"detail":"Not Found"},
which is correct FastAPI behaviour but a confusing first thing to meet.

The page is a single self-contained HTML file with inline CSS and JS -- no
build step, no bundler, no CDN. It talks to the same /chat/stream, /profile
and /health endpoints the CLI and voice clients use, so there is no separate
code path to keep in sync.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["web"])

INDEX = Path(__file__).resolve().parent.parent.parent / "web" / "index.html"
# The aim toolkit (sensitivity converter, crosshair builder, trainers, 3D
# range). Fully client-side, so it is served as-is.
AIM = INDEX.with_name("aim.html")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    if not INDEX.exists():  # pragma: no cover - only if the file is deleted
        return HTMLResponse(
            "<h1>Valorant Voice Coach</h1>"
            "<p>UI file missing. The API is fine: see "
            '<a href="/docs">/docs</a> and <a href="/health">/health</a>.</p>',
            status_code=200,
        )
    # Read per request rather than at import: editing the UI then refreshing
    # the browser is enough, no server restart.
    return HTMLResponse(
        INDEX.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@router.get("/aim", response_class=HTMLResponse, include_in_schema=False)
def aim() -> HTMLResponse:
    return HTMLResponse(
        AIM.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@router.get("/favicon.ico", include_in_schema=False)
def favicon() -> HTMLResponse:
    """Silences the 404 the browser logs on every page load."""
    return HTMLResponse(status_code=204)
