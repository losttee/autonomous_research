"""Pipeline orchestrator — wires the layers end-to-end.

Flow:  question -> LLM plan (parallelizable sub-tasks) -> execute batches
concurrently -> re-plan up to MAX_REPLAN times while evidence is thin ->
verify claim grounding -> synthesize -> report.

Sub-tasks within a dependency batch are independent, so each batch runs on a
thread pool (search + LLM calls are I/O-bound, so threads give real concurrency
here). The CostTracker is shared and thread-safe, so the budget cap still holds
across parallel workers.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from research_agent.core.config import get_settings
from research_agent.core.contracts import (
    FinalReport,
    Plan,
    SubTask,
    SubTaskResult,
    SubTaskStatus,
)
from research_agent.core.llm import LLMClient
from research_agent.core.logging import get_logger, log_step
from research_agent.executor.runner import run_subtask
from research_agent.executor.web_search import SearchTool, get_search_tool
from research_agent.guardrail.cost_tracker import BudgetExceeded, CostTracker
from research_agent.planner.planner import plan_question
from research_agent.synthesizer.report_generator import synthesize_llm
from research_agent.verifier.verifier import (
    dedupe_claims,
    flag_cross_contradictions,
    verify_results,
)

_log = get_logger("pipeline")

# Cap on concurrent workers per batch — a fuse against fanning out too wide.
_MAX_WORKERS = 5


@dataclass
class ResearchOutcome:
    """Report plus the plan and sub-task results behind it.

    Returned by run_research(..., return_details=True) so evaluation tooling
    can inspect claim-level data the FinalReport deliberately distills away.
    plan/results stay empty for reports served from memory recall.
    """

    report: FinalReport
    plan: Optional[Plan] = None
    results: list[SubTaskResult] = field(default_factory=list)


def _run_batch(
    batch: list[SubTask],
    tracker: CostTracker,
    tool: SearchTool,
    llm: Optional[LLMClient] = None,
) -> list[SubTaskResult]:
    """Run one dependency batch concurrently and return results in submission order."""
    if len(batch) == 1:
        return [run_subtask(batch[0], tracker, search_tool=tool, llm=llm)]
    workers = min(_MAX_WORKERS, len(batch))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda t: run_subtask(t, tracker, search_tool=tool, llm=llm), batch)
        )


def _execute_plan(
    plan: Plan,
    tracker: CostTracker,
    tool: SearchTool,
    llm: Optional[LLMClient] = None,
    on_progress: Optional[Callable[..., None]] = None,
) -> list[SubTaskResult]:
    """Execute all batches of a plan in dependency order; stop early on budget."""
    results: list[SubTaskResult] = []
    batches = plan.parallelizable_batches()
    for index, batch in enumerate(batches, 1):
        try:
            tracker.check()
        except BudgetExceeded:
            break  # return partial results gathered so far
        _emit(
            on_progress,
            "execute",
            f"Gathering sources (batch {index}/{len(batches)})…",
            batch=index,
            batches=len(batches),
            sub_tasks=len(batch),
        )
        results.extend(_run_batch(batch, tracker, tool, llm))
    return results


def _evidence_is_thin(results: list[SubTaskResult]) -> bool:
    """True when almost nothing was grounded — a signal to try one re-plan."""
    usable = [r for r in results if r.status == SubTaskStatus.DONE and r.claims]
    return len(usable) == 0


def _emit(
    on_progress: Optional[Callable[..., None]], step: str, msg: str, **extra: object
) -> None:
    """Fire one progress event. Best-effort: a broken listener never breaks a run."""
    if on_progress is None:
        return
    try:
        on_progress(step, msg, **extra)
    except Exception:
        pass


def _get_memory(use_memory: bool | None):
    """Return the memory store when enabled (arg overrides USE_MEMORY), else None."""
    enabled = get_settings().use_memory if use_memory is None else use_memory
    if not enabled:
        return None
    from research_agent.memory.store import get_memory_store

    return get_memory_store()


def _mark_recalled(report: FinalReport, score: float) -> FinalReport:
    """Flag a cached report transparently so a reused answer is never hidden."""
    note = f"Answer reused from a prior near-identical question (similarity {score:.2f})."
    report.recommendation = f"[recalled from memory] {report.recommendation}"
    if note not in report.uncertainties:
        report.uncertainties = [note, *report.uncertainties]
    log_step(
        _log,
        step_type="research_done",
        step_id=report.report_id,
        msg="served from memory",
        extra={"recalled": True, "score": round(score, 4)},
    )
    return report


def run_research(
    question: str,
    tracker: CostTracker | None = None,
    search_tool: SearchTool | None = None,
    allow_replan: bool = True,
    use_memory: bool | None = None,
    llm: Optional[LLMClient] = None,
    max_replan: int | None = None,
    return_details: bool = False,
    on_progress: Optional[Callable[..., None]] = None,
) -> FinalReport | ResearchOutcome:
    """Run the full pipeline for one question and return a FinalReport.

    Always returns a report — partial if the budget is exceeded mid-run. While
    the plan yields no grounded evidence, it re-plans up to MAX_REPLAN rounds
    (override with `max_replan`; 0 disables), each revision chained via
    previous_plan_id. `llm` overrides the shared client in every layer so tests
    can run the whole pipeline hermetically. `return_details=True` returns the
    plan and sub-task results alongside the report for evaluation tooling.
    `on_progress(step, msg, **extra)` receives one event per pipeline stage
    (plan/execute/replan/verify/synthesize/recall) for live UI updates.
    When memory is on, a near-identical past question short-circuits to the
    cached report; every fresh run's report + claims are remembered for next time.
    """
    tracker = tracker or CostTracker()
    tool = search_tool or get_search_tool()
    start = time.monotonic()

    memory = _get_memory(use_memory)
    if memory is not None:
        recalled = memory.recall_report(question, tracker=tracker)
        if recalled is not None:
            report, score = recalled
            _emit(
                on_progress,
                "recall",
                "Found a near-identical answer in memory…",
                score=round(score, 4),
            )
            recalled_report = _mark_recalled(report, score)
            if return_details:
                return ResearchOutcome(report=recalled_report)
            return recalled_report

    _emit(on_progress, "plan", "Planning the research…")
    plan = plan_question(question, tracker=tracker, llm=llm)
    results = _execute_plan(plan, tracker, tool, llm, on_progress)

    # Re-plan while nothing was grounded, up to the cap or until budget runs out.
    replan_cap = get_settings().max_replan if max_replan is None else max_replan
    if not allow_replan:
        replan_cap = 0
    while _evidence_is_thin(results) and plan.revision < replan_cap:
        try:
            tracker.check()
        except BudgetExceeded:
            break  # keep the thin results; the report will surface the gap
        _emit(
            on_progress,
            "replan",
            "Evidence is thin — re-planning…",
            revision=plan.revision + 1,
        )
        new_plan = plan_question(question, tracker=tracker, llm=llm)
        new_plan.revision = plan.revision + 1
        new_plan.previous_plan_id = plan.plan_id
        new_plan.replan_reason = f"revision {plan.revision} produced no grounded evidence"
        plan = new_plan
        results = _execute_plan(plan, tracker, tool, llm, on_progress)
        log_step(
            _log,
            step_type="replan",
            step_id=plan.plan_id,
            msg="re-planning after thin evidence",
            extra={"revision": plan.revision, "sub_tasks": len(plan.sub_tasks)},
        )

    # Verify grounding of every claim before synthesis — this is where claims
    # earn their confidence and contradictions surface.
    _emit(on_progress, "verify", "Checking each claim against its sources…")
    verify_results(results, tracker, llm=llm)

    # Consolidate before synthesis: merge near-duplicate findings (less token
    # spend, no repeated points), then flag findings that agree on the topic
    # but disagree on the numbers across sub-tasks.
    dedupe_claims(results)
    flag_cross_contradictions(results)

    _emit(on_progress, "synthesize", "Writing the report…")
    report = synthesize_llm(plan, results, tracker, llm=llm)

    # Remember this run so a future near-identical question can reuse it, and so
    # its verified claims are available as RAG sources next time.
    if memory is not None:
        try:
            memory.remember_report(report, tracker=tracker)
            for r in results:
                if r.status == SubTaskStatus.DONE:
                    memory.remember_claims(r, tracker=tracker)
        except Exception:  # best-effort persistence, never fail the run
            pass

    total_ms = int((time.monotonic() - start) * 1000)
    tracker.log_summary(step_id=report.report_id)
    log_step(
        _log,
        step_type="research_done",
        step_id=report.report_id,
        latency_ms=total_ms,
        cost_usd=tracker.snapshot().cost_usd,
        msg="research complete",
        extra={
            "sections": len(report.sections),
            "sources": len(report.all_sources),
            "revision": plan.revision,
        },
    )
    if return_details:
        return ResearchOutcome(report=report, plan=plan, results=results)
    return report
