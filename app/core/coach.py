"""The orchestrator.

Every request follows the same seven steps:

    1. domain gate          reject non-Valorant input before spending a token
    2. load profile         personalization data
    3. build query context   map / side / agents / rank band
    4. retrieve             hybrid search over the knowledge base
    5. build prompt         system + profile + notes + history + question
    6. generate             streamed from Ollama
    7. persist              transcript for short-term memory

Written as a plain class rather than an agent framework: there is no branching
decision for an agent loop to make here, and keeping the prompt visible is what
makes the system tunable.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

from sqlalchemy.orm import Session

from app.core import domain_gate, prompt
from app.core.filters import build_context
from app.db.repositories import ConversationRepo, ProfileRepo
from app.llm.base import LLMClient
from app.logging_setup import get_logger
from app.rag.retriever import Retriever
from app.rag.store import Chunk
from app.schemas.chat import ChatResponse, SourceOut

log = get_logger(__name__)

HISTORY_TURNS = 4


class CoachService:
    def __init__(self, llm: LLMClient, retriever: Retriever) -> None:
        self.llm = llm
        self.retriever = retriever

    # ------------------------------------------------------------- internals

    def _prepare(
        self, session: Session, username: str, message: str, session_id: str | None,
        use_rag: bool,
    ):
        """Steps 1-5. Shared by the buffered and streaming entry points."""
        t = {}
        t0 = time.perf_counter()

        repo = ProfileRepo(session)
        user = repo.get_or_create_user(username)
        profile = repo.get(username)
        convo = ConversationRepo(session)
        sid = session_id or uuid.uuid4().hex

        history = convo.recent(user.id, sid, limit=HISTORY_TURNS * 2)
        t["profile"] = (time.perf_counter() - t0) * 1000

        gate = domain_gate.check(message, has_history=bool(history))
        if not gate.allowed:
            log.info("gate.refused", reason=gate.reason, score=round(gate.score, 3))
            return None, sid, user, convo, profile, [], t, gate

        t1 = time.perf_counter()
        ctx = build_context(message, profile)
        chunks: list[Chunk] = self.retriever.search(ctx) if use_rag else []
        t["retrieval"] = (time.perf_counter() - t1) * 1000

        messages = prompt.build_messages(message, ctx, chunks, profile, history)
        return messages, sid, user, convo, profile, chunks, t, gate

    @staticmethod
    def _sources(chunks: list[Chunk]) -> list[SourceOut]:
        return [
            SourceOut(
                id=c.id, title=c.title, score=round(c.score, 4), map=c.map, side=c.side
            )
            for c in chunks
        ]

    # ----------------------------------------------------------- entry points

    async def answer(
        self,
        session: Session,
        username: str,
        message: str,
        session_id: str | None = None,
        use_rag: bool = True,
    ) -> ChatResponse:
        messages, sid, user, convo, _profile, chunks, t, gate = self._prepare(
            session, username, message, session_id, use_rag
        )
        if messages is None:
            return ChatResponse(
                reply=domain_gate.REFUSAL, sources=[], session_id=sid, refused=True
            )

        t2 = time.perf_counter()
        reply = (await self.llm.complete(messages)).strip()
        t["generation"] = (time.perf_counter() - t2) * 1000

        convo.add(user.id, sid, "user", message)
        convo.add(user.id, sid, "assistant", reply)

        return ChatResponse(
            reply=reply,
            sources=self._sources(chunks),
            session_id=sid,
            refused=False,
            timings_ms={k: round(v, 1) for k, v in t.items()},
        )

    async def stream(
        self,
        session: Session,
        username: str,
        message: str,
        session_id: str | None = None,
        use_rag: bool = True,
    ) -> AsyncIterator[str]:
        """Yield reply text incrementally.

        The voice pipeline consumes this generator and starts speaking at the
        first sentence boundary, which is what keeps perceived latency near
        one second instead of near six.
        """
        messages, sid, user, convo, _profile, _chunks, _t, _gate = self._prepare(
            session, username, message, session_id, use_rag
        )
        if messages is None:
            yield domain_gate.REFUSAL
            return

        parts: list[str] = []
        async for delta in self.llm.stream(messages):
            parts.append(delta)
            yield delta

        convo.add(user.id, sid, "user", message)
        convo.add(user.id, sid, "assistant", "".join(parts).strip())
