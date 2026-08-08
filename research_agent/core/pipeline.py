"""Pipeline orchestrator.

question -> plan (parallelizable sub-tasks) -> execute batches concurrently ->
re-plan while evidence is thin (up to MAX_REPLAN) -> verify claims ->
synthesize -> report.

Tasks in a batch are independent and I/O-bound, so each batch runs on a thread
pool; the CostTracker is thread-safe, so budget caps hold across workers.
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

# Max concurrent workers per batch.
_MAX_WORKERS = 5


@dataclass
class ResearchOutcome:
    """Report plus the plan and sub-task results behind it (for evaluation).

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
        except BudgetExceeded as exc:
            # Mark unstarted tasks SKIPPED so the report shows the gap.
            results.extend(
                SubTaskResult(
                    sub_task_id=task.sub_task_id,
                    status=SubTaskStatus.SKIPPED,
                    error=f"skipped due to budget: {exc.reason}",
                )
                for remaining in batches[index - 1:]
                for task in remaining
            )
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
    """True when nothing was grounded."""
    usable = [r for r in results if r.status == SubTaskStatus.DONE and r.claims]
    return len(usable) == 0


def _emit(
    on_progress: Optional[Callable[..., None]], step: str, msg: str, **extra: object
) -> None:
    """Best-effort progress event; a failing listener must not break the run."""
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
    """Mark a cached report so reuse is visible in the output."""
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
    verify_claims: bool = True,
) -> FinalReport | ResearchOutcome:
    """Run the full pipeline for one question and return a FinalReport.

    Always returns a report, partial if the budget runs out mid-run. Re-plans
    up to `max_replan` rounds (default MAX_REPLAN, 0 disables) while nothing is
    grounded. `llm` overrides the client in every layer (hermetic tests);
    `on_progress` gets one event per stage; `return_details=True` also returns
    plan/results; `verify_claims=False` skips verification (benchmark baseline).
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
            break  # out of budget; keep what we have
        _emit(
            on_progress,
            "replan",
            "Evidence is thin, re-planning…",
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

    # Verify claims before synthesis.
    if verify_claims:
        _emit(on_progress, "verify", "Checking each claim against its sources…")
        verify_results(results, tracker, llm=llm)

        # Merge near-duplicate claims, then flag number conflicts across tasks.
        dedupe_claims(results)
        flag_cross_contradictions(results)
    else:
        # Benchmark baseline: extractor claims pass through unverified.
        for result in results:
            for claim in result.claims:
                claim.supported = bool(claim.supporting_source_ids)

    _emit(on_progress, "synthesize", "Writing the report…")
    report = synthesize_llm(plan, results, tracker, llm=llm)

    # Store the report + claims for future recall/RAG.
    if memory is not None:
        try:
            memory.remember_report(report, tracker=tracker)
            for r in results:
                if r.status == SubTaskStatus.DONE:
                    memory.remember_claims(r, tracker=tracker)
        except Exception:  # best-effort
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
