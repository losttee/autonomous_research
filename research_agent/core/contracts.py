"""Shared data models for all layers: planner, executor, verifier, synthesizer.

Layers exchange only these models, so claims keep their source ids and
verifier output all the way into the final report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class SubTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class SourceType(str, Enum):
    WEB = "web"
    INTERNAL_RAG = "internal_rag"
    CALCULATOR = "calculator"
    MEMORY = "memory"


class SourceRef(BaseModel):
    """A retrieved source; the unit every citation points to."""

    source_id: str = Field(default_factory=lambda: _new_id("src"))
    type: SourceType
    title: str = ""
    url: Optional[str] = None
    snippet: str = Field(default="", description="Original text span used for grounding checks")
    retrieved_at: datetime = Field(default_factory=_utcnow)
    # None = not scored yet.
    reliability: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("url")
    @classmethod
    def _strip_url(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v


class Claim(BaseModel):
    """One factual assertion; the verifier fills supported/confidence."""

    claim_id: str = Field(default_factory=lambda: _new_id("clm"))
    text: str
    supporting_source_ids: list[str] = Field(default_factory=list)
    # None = not verified yet.
    supported: Optional[bool] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Set when sources conflict; surfaced in the final report.
    contradiction_note: Optional[str] = None

    @property
    def is_grounded(self) -> bool:
        return bool(self.supporting_source_ids) and self.supported is True


class SubTask(BaseModel):
    """A unit of work from the planner; independent tasks run in parallel."""

    sub_task_id: str = Field(default_factory=lambda: _new_id("task"))
    description: str
    # Ids of tasks that must finish first.
    depends_on: list[str] = Field(default_factory=list)
    tool_hint: Optional[SourceType] = Field(
        default=None, description="Hint for which tool to use; Executor may override"
    )


class SubTaskResult(BaseModel):
    """Result of one sub-task: distilled claims + sources, no raw tool output."""

    sub_task_id: str
    status: SubTaskStatus = SubTaskStatus.DONE
    claims: list[Claim] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: int = 0

    @property
    def mean_confidence(self) -> float:
        if not self.claims:
            return 0.0
        return sum(c.confidence for c in self.claims) / len(self.claims)


class PlannerStrategy(str, Enum):
    REACT = "react"
    PLAN_AND_EXECUTE = "plan_and_execute"


class Plan(BaseModel):
    """Plan from the planner. revision > 0 means it was re-planned;
    previous_plan_id chains the revisions."""

    plan_id: str = Field(default_factory=lambda: _new_id("plan"))
    question: str
    strategy: PlannerStrategy = PlannerStrategy.PLAN_AND_EXECUTE
    sub_tasks: list[SubTask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    revision: int = 0
    previous_plan_id: Optional[str] = None
    replan_reason: Optional[str] = None

    def parallelizable_batches(self) -> list[list[SubTask]]:
        """Group sub-tasks by dependency layer; each batch can run in parallel."""
        done: set[str] = set()
        batches: list[list[SubTask]] = []
        remaining = list(self.sub_tasks)
        while remaining:
            ready = [t for t in remaining if all(d in done for d in t.depends_on)]
            if not ready:  # cyclic/missing dependency; bail out instead of looping
                batches.append(remaining)
                break
            batches.append(ready)
            done.update(t.sub_task_id for t in ready)
            remaining = [t for t in remaining if t.sub_task_id not in done]
        return batches


class ConfidenceBand(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ReportSection(BaseModel):
    """A section of the final report, with citations and a confidence level."""

    heading: str
    body: str = Field(description="Text with inline [source_id] citations")
    cited_source_ids: list[str] = Field(default_factory=list)
    confidence_band: ConfidenceBand = ConfidenceBand.UNKNOWN


class FinalReport(BaseModel):
    """Final output: recommendation, cited sections, confidence, and explicit
    uncertainties/contradictions."""

    report_id: str = Field(default_factory=lambda: _new_id("rpt"))
    question: str
    plan_id: str
    recommendation: str = Field(description="Main conclusion / recommendation")
    sections: list[ReportSection] = Field(default_factory=list)
    all_sources: list[SourceRef] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainties: list[str] = Field(
        default_factory=list, description="What the agent is unsure of / could not find data for"
    )
    contradictions: list[str] = Field(
        default_factory=list, description="Conflicting sources that were detected"
    )
    created_at: datetime = Field(default_factory=_utcnow)
