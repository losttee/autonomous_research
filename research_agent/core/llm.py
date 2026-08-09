"""Shared LLM client. Layers call complete()/complete_json() instead of the
OpenAI SDK directly, and every call records usage into the CostTracker."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from openai import OpenAI

from research_agent.core.config import Settings, get_settings
from research_agent.core.logging import get_logger

if TYPE_CHECKING:
    from research_agent.guardrail.cost_tracker import CostTracker

_log = get_logger("core.llm")


class LLMError(Exception):
    """Raised when a model call fails."""


class LLMClient:
    """Thin wrapper over the OpenAI SDK, shared across all layers."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._client = OpenAI(
            api_key=self._settings.llm_api_key,
            base_url=self._settings.llm_base_url,
            timeout=float(self._settings.request_timeout_sec),
        )

    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        tracker: Optional["CostTracker"] = None,
        json_mode: bool = False,
    ) -> str:
        """One chat completion; returns the text. Records cost when a tracker is given.

        Raises LLMError on any failure; callers decide how to degrade.
        """
        model = model or self._settings.worker_model
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            _log.warning(
                "llm call failed",
                extra={"extra_fields": {"model": model, "error": str(exc)}},
            )
            raise LLMError(str(exc)) from exc

        usage = resp.usage
        if tracker is not None and usage is not None:
            tracker.record_llm_call(
                model, usage.prompt_tokens or 0, usage.completion_tokens or 0
            )

        content = resp.choices[0].message.content if resp.choices else None
        return content or ""

    def complete_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        tracker: Optional["CostTracker"] = None,
    ) -> Any:
        """Like complete() but parse the reply as JSON. Raises LLMError on bad JSON."""
        raw = self.complete(
            prompt,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            tracker=tracker,
            json_mode=True,
        )
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMError(f"model did not return valid JSON: {exc}") from exc


_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Singleton."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
