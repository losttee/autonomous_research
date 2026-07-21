"""Global configuration — read from environment variables / .env.

Model tier: the planner uses a strong (expensive) model only for planning/re-planning;
workers + verifier use a cheap model because they run many times. This is the main
cost lever: token spend explains most of the quality variance, so concentrate the
expensive model where it creates the most value (planning) and cheapen the repeated work.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Price in USD per 1M tokens (input, output). Update when pricing changes.
# Used by the cost_tracker to estimate cost before and during a run.
# NOTE: the current endpoint is an OpenAI-compatible gateway; pricing for the
# self-hosted models is unknown, so it is set to 0.0 (cost tracking will report $0
# for them). gpt-4o pricing kept for reference. Set real prices per deployment
# via the MODEL_PRICING env var (see Settings.model_pricing) instead of editing
# this table.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "llama-2-13b": (0.0, 0.0),
    "text-embedding-3-small": (0.02, 0.0),
}


@lru_cache(maxsize=1)
def _parse_pricing(raw: str) -> dict[str, tuple[float, float]]:
    """Parse "model:in_price:out_price,model:in_price:out_price" (USD/1M tokens).

    Malformed entries are skipped silently — a bad price line must never break
    a research run; that model just falls back to the built-in table.
    """
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
    # The gateway speaks the OpenAI API, so we use the OpenAI SDK with a custom
    # base_url. api_key + base_url are shared across all model roles here.
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

    # --- Cost ---
    # Overrides the MODEL_PRICING table above without a code change, format
    # "model:in_price:out_price,..." (USD per 1M tokens). Set real prices for
    # gateway models here so cost tracking reports real dollars.
    model_pricing: str = Field(default="", alias="MODEL_PRICING")

    # --- Web search tool ---
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    search_max_results: int = Field(default=5, alias="SEARCH_MAX_RESULTS")

    # --- Memory / RAG ---
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    use_memory: bool = Field(default=True, alias="USE_MEMORY")
    # High threshold: only reuse a past report when the question is near-identical,
    # so we don't serve a stale answer to a merely-similar question.
    memory_recall_threshold: float = Field(
        default=0.92, alias="MEMORY_RECALL_THRESHOLD"
    )

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")
    # JSON-lines file every pipeline step is appended to; the monitoring
    # dashboard aggregates from it. Empty string disables the file sink.
    log_file: str = Field(default="./data/logs/pipeline.jsonl", alias="LOG_FILE")

    def price_for(self, model: str) -> tuple[float, float]:
        """Return (input price, output price) USD/1M tokens; (0,0) if model unknown.

        The MODEL_PRICING env override wins over the built-in table.
        """
        overrides = _parse_pricing(self.model_pricing)
        if model in overrides:
            return overrides[model]
        return MODEL_PRICING.get(model, (0.0, 0.0))

    def estimate_cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_price, out_price = self.price_for(model)
        return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton — load once, reuse across the whole system."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
