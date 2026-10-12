"""Ollama client over the OpenAI-compatible endpoint.

Why /v1/chat/completions rather than Ollama's native /api/chat: the payload
and the SSE stream format are the industry standard, so switching providers
later is a base-URL change instead of a rewrite.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.config import settings
from app.llm.base import ChatMessage
from app.logging_setup import get_logger

log = get_logger(__name__)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.temperature = (
            settings.llm_temperature if temperature is None else temperature
        )
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout_s)

    def _payload(self, messages: list[ChatMessage], stream: bool) -> dict:
        # Note on num_ctx: it is deliberately NOT sent here. Ollama's
        # OpenAI-compatible endpoint silently discards an `options` block, so
        # sending it looks like it works while changing nothing (verified with
        # `ollama ps`, which kept reporting the default context). Context
        # length is a load-time property of the model anyway, so it belongs in
        # a Modelfile variant or in OLLAMA_CONTEXT_LENGTH on the server.
        # See scripts/setup_ollama.ps1.
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

    async def complete(self, messages: list[ChatMessage]) -> str:
        r = await self._client.post(
            f"{self.base_url}/chat/completions", json=self._payload(messages, False)
        )
        if r.status_code != 200:
            raise OllamaError(f"{r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Yield content deltas as they arrive.

        This generator is the backbone of the voice latency budget: the caller
        splits it into sentences and starts speaking before generation ends.
        """
        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=self._payload(messages, True),
        ) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="replace")
                raise OllamaError(f"{r.status_code}: {body[:300]}")
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self.base_url}/models")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
