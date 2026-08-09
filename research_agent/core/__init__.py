"""Core: data contracts, config, logging.

Does not re-export the pipeline or any layer, so this package stays cheap to
import and free of import cycles.
"""

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

__all__ = [
    "Settings",
    "get_settings",
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
