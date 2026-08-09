"""Verifier depth tests: claim dedup, cross-sub-task contradictions, and the
adversarial second pass. Deterministic (FakeLLM), no network, no API key.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_verifier_depth.py -v
"""

from __future__ import annotations

import json

from research_agent.core.contracts import (
    Claim,
    Plan,
    SourceRef,
    SourceType,
    SubTask,
    SubTaskResult,
    SubTaskStatus,
)
from research_agent.guardrail.cost_tracker import CostTracker
from research_agent.synthesizer.report_generator import synthesize
from research_agent.verifier.verifier import (
    adversarial_pass,
    dedupe_claims,
    find_cross_contradictions,
    flag_cross_contradictions,
    verify_results,
)


class QueueLLM:
    """Stand-in LLMClient that pops canned JSON replies in call order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def complete_json(self, prompt, *, model=None, system=None, max_tokens=1024,
                      temperature=0.2, tracker=None):
        self.calls += 1
        if tracker is not None:
            tracker.record_llm_call(model or "fake", 60, 20)
        return json.loads(self._replies.pop(0))


def _src(source_id: str = "src_aaa", reliability: float = 0.8) -> SourceRef:
    return SourceRef(
        type=SourceType.WEB, source_id=source_id,
        snippet="plan A costs $10 per month", reliability=reliability,
    )


def _result(claims: list[Claim], sources: list[SourceRef], task_id: str) -> SubTaskResult:
    return SubTaskResult(
        sub_task_id=task_id, status=SubTaskStatus.DONE,
        claims=claims, sources=sources,
    )


# --- dedup --------------------------------------------------------------------


def test_dedupe_merges_near_identical_claims() -> None:
    a = Claim(text="Plan A premium is $10 per month",
              supporting_source_ids=["src_1"], supported=True, confidence=0.8)
    b = Claim(text="Plan A premium is $10 per month.",
              supporting_source_ids=["src_2"], supported=True, confidence=0.6)
    results = [
        _result([a], [_src("src_1")], "t1"),
        _result([b], [_src("src_2")], "t2"),
    ]

    dedupe_claims(results)

    remaining = [c for r in results for c in r.claims]
    assert len(remaining) == 1, "near-identical claims collapse into one"
    survivor = remaining[0]
    assert survivor.supporting_source_ids == ["src_1", "src_2"], "sources union"
    assert survivor.confidence == 0.8, "the stronger verdict wins"


def test_dedupe_keeps_distinct_claims() -> None:
    a = Claim(text="Plan A premium is $10 per month",
              supporting_source_ids=["src_1"], supported=True, confidence=0.7)
    b = Claim(text="Plan B deductible is $500",
              supporting_source_ids=["src_2"], supported=True, confidence=0.7)
    results = [_result([a, b], [_src("src_1"), _src("src_2")], "t1")]

    dedupe_claims(results)
    assert len(results[0].claims) == 2


def test_dedupe_preserves_contradiction_notes() -> None:
    a = Claim(text="Plan A premium is $10 per month",
              supporting_source_ids=["src_1"], supported=True, confidence=0.8)
    b = Claim(text="Plan A premium is $10 per month!",
              supporting_source_ids=["src_2"], supported=True, confidence=0.5,
              contradiction_note="source says annual, not monthly")
    results = [
        _result([a], [_src("src_1")], "t1"),
        _result([b], [_src("src_2")], "t2"),
    ]

    dedupe_claims(results)
    remaining = [c for r in results for c in r.claims]
    assert len(remaining) == 1
    assert remaining[0].contradiction_note, "notes must survive the merge"


# --- cross-sub-task contradictions ---------------------------------------------


def _conflicting_results() -> list[SubTaskResult]:
    a = Claim(text="Plan A premium is $10 per month",
              supporting_source_ids=["src_1"], supported=True, confidence=0.8)
    b = Claim(text="Plan A premium is $15 per month",
              supporting_source_ids=["src_2"], supported=True, confidence=0.7)
    return [
        _result([a], [_src("src_1")], "t1"),
        _result([b], [_src("src_2")], "t2"),
    ]


def test_cross_contradiction_flags_conflicting_numbers() -> None:
    results = _conflicting_results()
    pairs = find_cross_contradictions(results)
    assert len(pairs) == 1


def test_cross_contradiction_ignores_matching_numbers() -> None:
    a = Claim(text="Plan A premium is $10 per month",
              supporting_source_ids=["src_1"], supported=True, confidence=0.8)
    b = Claim(text="The premium of plan A is $10 monthly",
              supporting_source_ids=["src_2"], supported=True, confidence=0.7)
    results = [
        _result([a], [_src("src_1")], "t1"),
        _result([b], [_src("src_2")], "t2"),
    ]
    assert find_cross_contradictions(results) == []


def test_cross_contradiction_ignores_unsupported_claims() -> None:
    results = _conflicting_results()
    results[1].claims[0].supported = False
    assert find_cross_contradictions(results) == []


def test_flagged_conflicts_surface_in_the_report() -> None:
    results = _conflicting_results()
    flagged = flag_cross_contradictions(results)
    assert flagged == 1

    plan = Plan(question="premium?", sub_tasks=[
        SubTask(sub_task_id="t1", description="premium via source 1"),
        SubTask(sub_task_id="t2", description="premium via source 2"),
    ])
    report = synthesize(plan, results)

    # Both sides of the disagreement must appear in the report.
    assert len(report.contradictions) == 2
    joined = " | ".join(report.contradictions)
    assert "$15" in joined and "$10" in joined


# --- adversarial pass -----------------------------------------------------------

_ENTAILS = '{"supported": true, "confidence": 0.9, "contradiction": ""}'


def test_adversarial_retracts_a_refuted_claim() -> None:
    claim = Claim(text="Plan A costs $12 per month",
                  supporting_source_ids=["src_aaa"], supported=True, confidence=0.8)
    results = [_result([claim], [_src()], "t1")]
    llm = QueueLLM([
        '{"refuted": true, "reason": "snippet says $10 per month, not $12"}',
    ])

    refuted = adversarial_pass(results, CostTracker(), llm=llm)

    assert refuted == 1 and llm.calls == 1
    assert claim.supported is False
    assert claim.confidence == 0.0
    assert claim.contradiction_note and "refuted" in claim.contradiction_note


def test_adversarial_keeps_a_claim_that_survives() -> None:
    claim = Claim(text="Plan A costs $10 per month",
                  supporting_source_ids=["src_aaa"], supported=True, confidence=0.8)
    results = [_result([claim], [_src()], "t1")]
    llm = QueueLLM(['{"refuted": false, "reason": ""}'])

    refuted = adversarial_pass(results, CostTracker(), llm=llm)

    assert refuted == 0
    assert claim.supported is True and claim.confidence == 0.8


def test_adversarial_skips_weak_claims() -> None:
    claim = Claim(text="Plan A costs $10 per month",
                  supporting_source_ids=["src_aaa"], supported=True, confidence=0.4)
    results = [_result([claim], [_src()], "t1")]
    llm = QueueLLM([])

    assert adversarial_pass(results, CostTracker(), llm=llm) == 0
    assert llm.calls == 0, "below-threshold claims never trigger a call"


def test_verify_results_runs_adversarial_only_when_enabled() -> None:
    def fresh():
        claim = Claim(text="Plan A costs $10 per month",
                      supporting_source_ids=["src_aaa"])
        return [_result([claim], [_src()], "t1")], claim

    # Enabled: entailment call + adversarial call.
    results, claim = fresh()
    llm = QueueLLM([_ENTAILS, '{"refuted": false, "reason": ""}'])
    verify_results(results, CostTracker(), llm=llm, adversarial=True)
    assert llm.calls == 2

    # Disabled: only the entailment call (default setting is off).
    results, claim = fresh()
    llm = QueueLLM([_ENTAILS])
    verify_results(results, CostTracker(), llm=llm, adversarial=False)
    assert llm.calls == 1
    assert claim.supported is True


if __name__ == "__main__":
    test_dedupe_merges_near_identical_claims()
    test_dedupe_keeps_distinct_claims()
    test_dedupe_preserves_contradiction_notes()
    test_cross_contradiction_flags_conflicting_numbers()
    test_cross_contradiction_ignores_matching_numbers()
    test_cross_contradiction_ignores_unsupported_claims()
    test_flagged_conflicts_surface_in_the_report()
    test_adversarial_retracts_a_refuted_claim()
    test_adversarial_keeps_a_claim_that_survives()
    test_adversarial_skips_weak_claims()
    test_verify_results_runs_adversarial_only_when_enabled()
    print("VERIFIER DEPTH TESTS PASSED")
