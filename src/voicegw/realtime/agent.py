"""Pluggable agent hook.

The realtime loop hands a final transcript to an async callable and consumes
text deltas. The default implementation talks to any OpenAI-compatible
``/chat/completions`` endpoint, so you can point this at your existing agent
(Ollama, vLLM, LM Studio, OpenAI, your own FastAPI service) purely via config.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx

from ..common.config import Settings, get_settings

log = logging.getLogger(__name__)

#: ``(user_text, history) -> async iterator of text deltas``
AgentHook = Callable[[str, list[dict]], AsyncIterator[str]]


class EchoAgent:
    """No-LLM fallback: speaks the transcript back. Useful for testing the loop."""

    async def __call__(self, text: str, history: list[dict]) -> AsyncIterator[str]:
        yield text


class OpenAICompatibleAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.agent_base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {self.settings.agent_api_key}"},
                timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __call__(self, text: str, history: list[dict]) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": self.settings.agent_system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": text})

        payload = {"model": self.settings.agent_model, "messages": messages, "stream": True}
        try:
            async with self._http().stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            log.error("agent request failed: %s", exc)
            yield "عذرًا، تعذّر الوصول إلى المساعد الآن."


def build_agent(settings: Settings | None = None) -> AgentHook | Awaitable:
    settings = settings or get_settings()
    if not settings.agent_base_url:
        return EchoAgent()
    return OpenAICompatibleAgent(settings)
