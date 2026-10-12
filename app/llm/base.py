"""LLM client contract.

The orchestrator depends on this protocol, never on Ollama directly. Any
OpenAI-compatible backend (a bigger local model, a hosted API) can be dropped
in later without touching app/core.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, TypedDict


class ChatMessage(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(Protocol):
    async def complete(self, messages: list[ChatMessage]) -> str: ...

    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...

    async def health(self) -> bool: ...
