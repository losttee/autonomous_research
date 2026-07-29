"""Evaluation harness tests — metrics logic and pricing overrides.

Run: .venv/Scripts/python.exe -m pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

import json

from research_agent.core.config import get_settings
from research_agent.core.contracts import (
    Claim,
    FinalReport,
    ReportSection,
    SourceRef,
    SourceType,
    SubTaskResult,
    SubTaskStatus,
)
from research_agent.guardrail.cost_tracker import CostTracker

from evaluation.metrics import (
    citation_integrity,
    collect_supported_claims,
    grounding_precision,
    honesty_ok,
    keyword_hits,
)


def _report(rec_extra: str = "ok", body_extra: str = "") -> FinalReport:
    src = SourceRef(type=SourceType.WEB, source_id="src_aaa", snippet="x")
    return FinalReport(
        question="q",
        plan_id="p",
        recommendation=f"rec {rec_extra}",
        sections=[ReportSection(heading="h", body=body_extra or "body")],
        all_sources=[src],
        overall_confidence=0.7,
    )


def test_citation_integrity_flags_unresolved() -> None:
    # well-formed id shape (hex) but maps to no real source
    report = _report(body_extra="real [src_aaa] and ghost [src_deadbeef01]")
    resolved, has_cites, unresolved = citation_integrity(report)
    assert has_cites and not resolved
    assert unresolved == ["src_deadbeef01"]


def test_citation_integrity_passes_when_all_resolve() -> None:
    report = _report(body_extra="also cites [src_aaa]")
    resolved, has_cites, unresolved = citation_integrity(report)
    assert resolved and has_cites and unresolved == []


def test_citation_integrity_reports_uncited_reports() -> None:
    resolved, has_cites, _ = citation_integrity(_report())
    assert resolved and not has_cites, "no citations is vacuously clean, flagged via has_citations"


def test_keyword_hits_are_case_insensitive() -> None:
    report = _report(rec_extra="The answer is CANBERRA today")
    assert keyword_hits(report, ["canberra", "paris"]) == (1, 2)


def test_honesty_passes_on_uncertainty() -> None:
    report = _report()
    report.uncertainties = ["no data for X"]
    assert honesty_ok(report)


def test_honesty_passes_on_low_confidence() -> None:
    report = _report()
    report.overall_confidence = 0.3
    assert honesty_ok(report)


def test_honesty_fails_on_confident_silence() -> None:
    report = _report()
    report.overall_confidence = 0.9
    assert not honesty_ok(report)


class JudgeLLM:
    """Replies from a queue so each judged claim can receive its own verdict."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def complete_json(self, prompt, **kwargs):
        self.calls += 1
        tracker = kwargs.get("tracker")
        if tracker is not None:
            tracker.record_llm_call("fake", 50, 20)
        return json.loads(self._replies.pop(0))


def _supported_pairs():
    src = SourceRef(
        type=SourceType.WEB, source_id="src_aaa",
        snippet="plan A costs $10", reliability=0.8,
    )
    claims = [
        Claim(text="Plan A costs $10", supporting_source_ids=["src_aaa"],
              supported=True, confidence=0.8),
        Claim(text="Plan A is free", supporting_source_ids=["src_aaa"],
              supported=True, confidence=0.6),
        Claim(text="unverified", supporting_source_ids=["src_aaa"],
              supported=False),
    ]
    result = SubTaskResult(
        sub_task_id="t1", status=SubTaskStatus.DONE, claims=claims, sources=[src]
    )
    return collect_supported_claims([result])


def test_grounding_precision_judges_supported_claims_only() -> None:
    pairs = _supported_pairs()
    assert len(pairs) == 2, "unsupported claims must never be sampled"
    judge = JudgeLLM([
        '{"grounded": true, "reason": "ok"}',
        '{"grounded": false, "reason": "snippet says the opposite"}',
    ])
    score, sampled = grounding_precision(pairs, judge, CostTracker(), sample_n=10)
    assert sampled == 2 and judge.calls == 2
    assert abs(score - 0.5) < 1e-6


def test_grounding_precision_empty_without_supported_claims() -> None:
    score, sampled = grounding_precision([], JudgeLLM([]), CostTracker())
    assert score is None and sampled == 0


def test_pricing_override_from_env_format() -> None:
    settings = get_settings().model_copy()
    settings.model_pricing = "gpt-4o:1.0:2.0,broken_entry,llama-2-13b:0.1:x"
    assert settings.price_for("gpt-4o") == (1.0, 2.0)
    # malformed entries fall back to the built-in table (llama: 0.0/0.0)
    assert settings.price_for("llama-2-13b") == (0.0, 0.0)
    assert settings.price_for("unknown-model") == (0.0, 0.0)


if __name__ == "__main__":
    test_citation_integrity_flags_unresolved()
    test_citation_integrity_passes_when_all_resolve()
    test_citation_integrity_reports_uncited_reports()
    test_keyword_hits_are_case_insensitive()
    test_honesty_passes_on_uncertainty()
    test_honesty_passes_on_low_confidence()
    test_honesty_fails_on_confident_silence()
    test_grounding_precision_judges_supported_claims_only()
    test_grounding_precision_empty_without_supported_claims()
    test_pricing_override_from_env_format()
    print("EVALUATION TESTS PASSED")
