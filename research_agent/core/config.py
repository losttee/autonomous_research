"""Settings loaded from environment variables / .env.

The planner runs on the strong model; workers/verifier run on the cheap one,
since they are called many times per request.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# USD per 1M tokens (input, output), used for cost estimates. Gateway model
# prices vary per deployment; override them with the MODEL_PRICING env var.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "llama-2-13b": (0.0, 0.0),
    "text-embedding-3-small": (0.02, 0.0),
}


@lru_cache(maxsize=1)
def _parse_pricing(raw: str) -> dict[str, tuple[float, float]]:
    """Parse "model:in:out,model:in:out" price overrides; skips bad entries."""
    out: dict[str, tuple[float, float]] = {}
    for chunk in raw.split(","):
        parts = [p.strip() for p in chunk.split(":")]
        if len(parts) != 3 or not parts[0]:
            continue
        try:
            out[parts[0]] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue
    return out


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM (OpenAI-compatible endpoint) ---
    # OpenAI-compatible gateway; one key/base_url for every model role.
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.pinkyne.com/v1", alias="LLM_BASE_URL")
    planner_model: str = Field(default="gpt-4o", alias="PLANNER_MODEL")
    worker_model: str = Field(default="gpt-4o", alias="WORKER_MODEL")
    verifier_model: str = Field(default="gpt-4o", alias="VERIFIER_MODEL")
    synth_model: str = Field(default="gpt-4o", alias="SYNTH_MODEL")

    # --- Guardrail (hard limits for a single request) ---
    max_llm_calls: int = Field(default=25, alias="MAX_LLM_CALLS")
    max_tool_calls: int = Field(default=40, alias="MAX_TOOL_CALLS")
    max_tokens_budget: int = Field(default=200_000, alias="MAX_TOKENS_BUDGET")
    max_usd_budget: float = Field(default=1.5, alias="MAX_USD_BUDGET")
    request_timeout_sec: int = Field(default=180, alias="REQUEST_TIMEOUT_SEC")
    max_replan: int = Field(default=3, alias="MAX_REPLAN")
    # Extra verifier round that tries to refute strong claims. Costs more.
    adversarial_verify: bool = Field(default=False, alias="ADVERSARIAL_VERIFY")

    # --- Cost ---
    # Price overrides, format "model:in:out,..." (USD per 1M tokens).
    model_pricing: str = Field(default="", alias="MODEL_PRICING")

    # --- Web search tool ---
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    search_max_results: int = Field(default=5, alias="SEARCH_MAX_RESULTS")

    # --- Tools ---
    # Local folder the document tool searches for INTERNAL_RAG sources.
    document_root: str = Field(default="./data/documents", alias="DOCUMENT_ROOT")

    # --- Memory / RAG ---
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    use_memory: bool = Field(default=True, alias="USE_MEMORY")
    # Only reuse a cached report when the question is near-identical.
    memory_recall_threshold: float = Field(
        default=0.92, alias="MEMORY_RECALL_THRESHOLD"
    )
    # Memories older than this are dropped. 0 disables expiry.
    memory_ttl_days: int = Field(default=30, alias="MEMORY_TTL_DAYS")

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")
    # JSON-lines log the monitoring dashboard reads. Empty disables it.
    log_file: str = Field(default="./data/logs/pipeline.jsonl", alias="LOG_FILE")

    def price_for(self, model: str) -> tuple[float, float]:
        """(input, output) price in USD/1M tokens; env override wins, else table."""
        overrides = _parse_pricing(self.model_pricing)
        if model in overrides:
            return overrides[model]
        return MODEL_PRICING.get(model, (0.0, 0.0))

    def estimate_cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_price, out_price = self.price_for(model)
        return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
