"""Data-access helpers.

Route handlers and the orchestrator call these instead of writing queries, so
swapping the storage engine never leaks upward.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Message, Profile, User
from app.schemas.profile import ProfileIn, ProfileOut


def _join(values: list[str]) -> str:
    return "\n".join(v.strip() for v in values if v and v.strip())


def _split(value: str) -> list[str]:
    return [v.strip() for v in (value or "").splitlines() if v.strip()]


class ProfileRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def get_or_create_user(self, username: str) -> User:
        user = self.s.scalar(select(User).where(User.username == username))
        if user is None:
            user = User(username=username)
            user.profile = Profile()
            self.s.add(user)
            self.s.commit()
            self.s.refresh(user)
        elif user.profile is None:
            # Repairs rows created before the profile relationship existed.
            user.profile = Profile()
            self.s.commit()
        return user

    def get(self, username: str) -> ProfileOut:
        user = self.get_or_create_user(username)
        p = user.profile
        return ProfileOut(
            user_id=user.id,
            username=user.username,
            current_rank=p.current_rank,
            target_rank=p.target_rank,
            preferred_role=p.preferred_role,
            playstyle=p.playstyle,  # type: ignore[arg-type]
            main_agents=_split(p.main_agents),
            favorite_maps=_split(p.favorite_maps),
            strengths=_split(p.strengths),
            weaknesses=_split(p.weaknesses),
            notes=p.notes,
            dpi=p.dpi or 0,
            sensitivity=p.sensitivity or 0.0,
        )

    def upsert(self, username: str, data: ProfileIn) -> ProfileOut:
        user = self.get_or_create_user(username)
        p = user.profile
        p.current_rank = data.current_rank
        p.target_rank = data.target_rank
        p.preferred_role = data.preferred_role
        p.playstyle = data.playstyle
        p.main_agents = _join(data.main_agents)
        p.favorite_maps = _join(data.favorite_maps)
        p.strengths = _join(data.strengths)
        p.weaknesses = _join(data.weaknesses)
        p.notes = data.notes
        p.dpi = data.dpi
        p.sensitivity = data.sensitivity
        self.s.commit()
        return self.get(username)


class ConversationRepo:
    """Short-term memory: the last few turns, replayed into the prompt.

    Capped by turn count rather than tokens because the whole prompt budget is
    small (see LLM_NUM_CTX) and retrieved context must stay the larger share.
    """

    def __init__(self, session: Session) -> None:
        self.s = session

    def add(self, user_id: str, session_id: str, role: str, content: str) -> None:
        self.s.add(
            Message(
                user_id=user_id, session_id=session_id, role=role, content=content
            )
        )
        self.s.commit()

    def recent(
        self, user_id: str, session_id: str, limit: int = 6
    ) -> list[tuple[str, str]]:
        rows = self.s.scalars(
            select(Message)
            .where(Message.user_id == user_id, Message.session_id == session_id)
            .order_by(Message.id.desc())
            .limit(limit)
        ).all()
        return [(m.role, m.content) for m in reversed(rows)]
