"""Smoke test — verify the core foundation runs.

Run: py demo.py
Confirms: (1) contracts/config/logger import OK, (2) a Plan -> SubTaskResult
-> FinalReport pipeline can be built, (3) one JSON log line with all
measurement fields is printed.
"""

from __future__ import annotations

from research_agent.core import (
    Claim,
    FinalReport,
    Plan,
    ReportSection,
    SourceRef,
    SourceType,
    SubTask,
    SubTaskResult,
    get_logger,
    get_settings,
    log_step,
)
from research_agent.core.contracts import ConfidenceBand


def main() -> None:
    settings = get_settings()
    log = get_logger("demo")

    # 1) Build a minimal data pipeline to check the contracts fit together.
    plan = Plan(
        question="Compare insurance plans A and B by premium and benefits.",
        sub_tasks=[
            SubTask(description="Look up premium of plan A", tool_hint=SourceType.WEB),
            SubTask(description="Look up premium of plan B", tool_hint=SourceType.WEB),
        ],
    )

    src = SourceRef(
        type=SourceType.WEB,
        title="Insurance plan A premium table",
        url="https://example.com/a",
        snippet="Plan A premium is 5M/year.",
        reliability=0.8,
    )
    claim = Claim(
        text="Plan A premium is 5M/year.",
        supporting_source_ids=[src.source_id],
        supported=True,
        confidence=0.82,
    )
    result = SubTaskResult(
        sub_task_id=plan.sub_tasks[0].sub_task_id,
        claims=[claim],
        sources=[src],
        tokens_used=1200,
        latency_ms=850,
    )

    report = FinalReport(
        question=plan.question,
        plan_id=plan.plan_id,
        recommendation="Not enough data to recommend yet — only plan A premium is known.",
        sections=[
            ReportSection(
                heading="Premium",
                body=f"Plan A: 5M/year [{src.source_id}].",
                cited_source_ids=[src.source_id],
                confidence_band=ConfidenceBand.MEDIUM,
            )
        ],
        all_sources=[src],
        overall_confidence=result.mean_confidence,
        uncertainties=["Could not look up plan B premium."],
    )

    # 2) Check a few contract invariants.
    batches = plan.parallelizable_batches()
    assert len(batches) == 1 and len(batches[0]) == 2, "2 independent sub-tasks -> 1 parallel batch"
    assert claim.is_grounded, "claim with source + supported=True -> grounded"
    assert 0.0 <= report.overall_confidence <= 1.0

    # 3) Print one JSON log line with all measurement fields.
    cost = settings.estimate_cost_usd(settings.worker_model, 1000, 200)
    log_step(
        log,
        step_type="smoke_test",
        step_id=result.sub_task_id,
        tokens=result.tokens_used,
        latency_ms=result.latency_ms,
        cost_usd=cost,
        msg="Foundation check OK",
        extra={
            "parallel_batches": len(batches),
            "claims": len(result.claims),
            "planner_model": settings.planner_model,
            "worker_model": settings.worker_model,
        },
    )

    print("\nSMOKE TEST PASSED")
    print(f"  Plan {plan.plan_id}: {len(plan.sub_tasks)} sub-tasks, {len(batches)} parallel batch")
    print(f"  Report {report.report_id}: confidence={report.overall_confidence:.2f}, "
          f"{len(report.uncertainties)} open uncertainties")


if __name__ == "__main__":
    main()
