"""Core: data contracts, config, logging — the foundation of every layer."""

from research_agent.core.config import Settings, get_settings
from research_agent.core.contracts import (
    Claim,
    ConfidenceBand,
    FinalReport,
    Plan,
    PlannerStrategy,
    ReportSection,
    SourceRef,
    SourceType,
    SubTask,
    SubTaskResult,
    SubTaskStatus,
)
from research_agent.core.embeddings import EmbeddingClient, get_embedding_client
from research_agent.core.logging import get_logger, log_step
from research_agent.core.pipeline import run_research
from research_agent.planner import plan_question

__all__ = [
    "Settings",
    "get_settings",
    "run_research",
    "plan_question",
    "EmbeddingClient",
    "get_embedding_client",
    "Claim",
    "ConfidenceBand",
    "FinalReport",
    "Plan",
    "PlannerStrategy",
    "ReportSection",
    "SourceRef",
    "SourceType",
    "SubTask",
    "SubTaskResult",
    "SubTaskStatus",
    "get_logger",
    "log_step",
]
