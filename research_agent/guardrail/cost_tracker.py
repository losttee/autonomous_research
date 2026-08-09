"""Guardrail: cost + budget tracking for a single research request.

Enforces hard caps on LLM calls, tool calls, tokens, USD and wall-clock time.
When a cap is exceeded, callers stop and return partial results.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from research_agent.core.config import Settings, get_settings
from research_agent.core.logging import get_logger, log_step


class BudgetExceeded(Exception):
    """Raised when a hard cap is hit."""

    def __init__(self, reason: str, snapshot: "CostSnapshot") -> None:
        super().__init__(reason)
        self.reason = reason
        self.snapshot = snapshot


@dataclass(frozen=True)
class CostSnapshot:
    """Immutable view of current spend."""

    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    elapsed_sec: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostTracker:
    """Accumulates spend across one request and enforces hard caps.

    Usage:
        tracker = CostTracker()
        tracker.check()                       # before each step; raises if over budget
        tracker.record_llm_call(model, in_tok, out_tok)
        tracker.record_tool_call()
    """

    settings: Settings = field(default_factory=get_settings)
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    _start: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self._log = get_logger("guardrail")
        # Sub-tasks run in parallel batches, so multiple threads mutate the
        # counters and read them via check(). Guard all mutation + snapshot.
        self._lock = threading.Lock()

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            return CostSnapshot(
                llm_calls=self.llm_calls,
                tool_calls=self.tool_calls,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cost_usd=round(self.cost_usd, 6),
                elapsed_sec=round(time.monotonic() - self._start, 3),
            )

    def record_llm_call(self, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.llm_calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cost_usd += self.settings.estimate_cost_usd(
                model, input_tokens, output_tokens
            )

    def record_tool_call(self) -> None:
        with self._lock:
            self.tool_calls += 1

    def check(self) -> None:
        """Raise BudgetExceeded if any hard cap is hit. Call before each costly step."""
        s = self.settings
        snap = self.snapshot()  # consistent read under lock
        reason: str | None = None
        if snap.llm_calls >= s.max_llm_calls:
            reason = f"max_llm_calls reached ({snap.llm_calls}/{s.max_llm_calls})"
        elif snap.tool_calls >= s.max_tool_calls:
            reason = f"max_tool_calls reached ({snap.tool_calls}/{s.max_tool_calls})"
        elif snap.total_tokens >= s.max_tokens_budget:
            reason = f"max_tokens_budget reached ({snap.total_tokens}/{s.max_tokens_budget})"
        elif snap.cost_usd >= s.max_usd_budget:
            reason = f"max_usd_budget reached (${snap.cost_usd:.4f}/${s.max_usd_budget})"
        elif snap.elapsed_sec >= s.request_timeout_sec:
            reason = f"request_timeout reached ({snap.elapsed_sec}s/{s.request_timeout_sec}s)"
        if reason is not None:
            log_step(
                self._log,
                step_type="budget_exceeded",
                step_id="guardrail",
                tokens=snap.total_tokens,
                latency_ms=int(snap.elapsed_sec * 1000),
                cost_usd=snap.cost_usd,
                msg=reason,
                extra={"llm_calls": snap.llm_calls, "tool_calls": snap.tool_calls},
            )
            raise BudgetExceeded(reason, snap)

    def log_summary(self, step_id: str = "request") -> None:
        snap = self.snapshot()
        log_step(
            self._log,
            step_type="cost_summary",
            step_id=step_id,
            tokens=snap.total_tokens,
            latency_ms=int(snap.elapsed_sec * 1000),
            cost_usd=snap.cost_usd,
            msg="request cost summary",
            extra={"llm_calls": snap.llm_calls, "tool_calls": snap.tool_calls},
        )
