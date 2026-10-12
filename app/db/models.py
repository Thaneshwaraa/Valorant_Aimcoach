"""ORM models.

Deliberately small for the MVP: a user, their coaching profile, and a
conversation transcript. The transcript is what later enables recalling what
was worked on in a previous session, with no extra schema work.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    profile: Mapped[Profile | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    messages: Mapped[list[Message]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Profile(Base):
    """The personalization payload injected into every prompt.

    List-valued fields (main_agents, strengths, ...) are stored as
    newline-joined text rather than a JSON column, so the schema ports to
    PostgreSQL without any dialect-specific types.
    """

    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )

    current_rank: Mapped[str] = mapped_column(String(32), default="unranked")
    target_rank: Mapped[str] = mapped_column(String(32), default="")
    preferred_role: Mapped[str] = mapped_column(String(32), default="")
    playstyle: Mapped[str] = mapped_column(String(32), default="balanced")

    main_agents: Mapped[str] = mapped_column(Text, default="")
    favorite_maps: Mapped[str] = mapped_column(Text, default="")
    strengths: Mapped[str] = mapped_column(Text, default="")
    weaknesses: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")

    # Mouse setup. 0 means "not set". eDPI is derived, never stored.
    dpi: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    sensitivity: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, onupdate=_now
    )

    user: Mapped[User] = relationship(back_populates="profile")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    session_id: Mapped[str] = mapped_column(String(32), index=True, default=_uuid)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)

    user: Mapped[User] = relationship(back_populates="messages")
