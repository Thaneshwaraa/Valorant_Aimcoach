"""Chat request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    username: str = Field(default="default", max_length=64)
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    # Set false to compare grounded vs. ungrounded answers while tuning the KB.
    use_rag: bool = True


class SourceOut(BaseModel):
    id: str
    title: str
    score: float
    map: str = ""
    side: str = ""


class ChatResponse(BaseModel):
    reply: str
    sources: list[SourceOut] = []
    session_id: str
    refused: bool = False
    timings_ms: dict[str, float] = {}
