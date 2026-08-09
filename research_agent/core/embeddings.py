"""Embedding client for the memory layer. When the API is unavailable it falls
back to a deterministic hashing embedding, so memory still works offline."""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING, Optional

from openai import OpenAI

from research_agent.core.config import Settings, get_settings
from research_agent.core.logging import get_logger

if TYPE_CHECKING:
    from research_agent.guardrail.cost_tracker import CostTracker

_log = get_logger("core.embeddings")

# Small on purpose: only needs to separate near-identical from unrelated text.
_HASH_DIM = 256

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_embedding(text: str, dim: int = _HASH_DIM) -> list[float]:
    """Offline fallback: bag-of-hashed-tokens, L2-normalized."""
    vec = [0.0] * dim
    for tok in _tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class EmbeddingClient:
    """Thin wrapper over the OpenAI embeddings API, shared across the memory layer."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._model = self._settings.embedding_model
        self._client = OpenAI(
            api_key=self._settings.llm_api_key,
            base_url=self._settings.llm_base_url,
            timeout=float(self._settings.request_timeout_sec),
        )
        # After one failure, stay on the fallback for the rest of the run.
        self._degraded = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def embed(
        self,
        texts: list[str],
        tracker: Optional["CostTracker"] = None,
    ) -> list[list[float]]:
        """Embed a batch of texts; falls back to the hash embedding on API failure."""
        if not texts:
            return []
        if not self._degraded and self._settings.llm_api_key:
            try:
                resp = self._client.embeddings.create(model=self._model, input=texts)
                if tracker is not None and getattr(resp, "usage", None) is not None:
                    tracker.record_llm_call(
                        self._model, resp.usage.prompt_tokens or 0, 0
                    )
                return [item.embedding for item in resp.data]
            except Exception as exc:
                self._degraded = True
                _log.warning(
                    "embedding call failed; using offline hashing fallback",
                    extra={"extra_fields": {"model": self._model, "error": str(exc)}},
                )
        return [_hash_embedding(t) for t in texts]

    def embed_one(
        self,
        text: str,
        tracker: Optional["CostTracker"] = None,
    ) -> list[float]:
        return self.embed([text], tracker=tracker)[0]


_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """Singleton."""
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client
