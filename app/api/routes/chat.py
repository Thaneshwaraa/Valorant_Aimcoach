"""Chat endpoints: buffered and streaming."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_coach, get_session
from app.core.coach import CoachService
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    session: Session = Depends(get_session),
    coach: CoachService = Depends(get_coach),
) -> ChatResponse:
    return await coach.answer(
        session=session,
        username=req.username,
        message=req.message,
        session_id=req.session_id,
        use_rag=req.use_rag,
    )


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    session: Session = Depends(get_session),
    coach: CoachService = Depends(get_coach),
) -> StreamingResponse:
    """Plain text/event-stream of content deltas.

    Kept as raw text rather than JSON envelopes so a terminal client can pipe
    it straight to the screen and the voice client straight to the sentence
    splitter.
    """

    async def gen():
        async for delta in coach.stream(
            session=session,
            username=req.username,
            message=req.message,
            session_id=req.session_id,
            use_rag=req.use_rag,
        ):
            yield delta

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
