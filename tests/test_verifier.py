"""Verifier tests: LLM entailment, reliability blending,
and contradiction detection.

Uses a FakeLLM so tests are deterministic and need no API key or network.
Run: .venv/Scripts/python.exe -m pytest tests/test_verifier.py -v
"""

from __future__ import annotations

import json

from research_agent.core.contracts import (
    Claim,
    SourceRef,
    SourceType,
    SubTaskResult,
    SubTaskStatus,
)
from research_agent.core.llm import LLMError
from research_agent.guardrail.cost_tracker import CostTracker
from research_agent.synthesizer.report_generator import synthesize
from research_agent.core.contracts import Plan, SubTask
from research_agent.verifier.verifier import verify_claim, verify_results


class FakeLLM:
    """Stand-in LLMClient: returns a canned JSON reply and records a fake cost."""

    def __init__(self, reply: str, *, raise_error: bool = False) -> None:
        self._reply = reply
        self._raise = raise_error
        self.calls = 0

    def complete_json(self, prompt, *, model=None, system=None, max_tokens=1024,
                      temperature=0.2, tracker=None):
        self.calls += 1
        if tracker is not None:
            tracker.record_llm_call(model or "fake", 100, 30)
        if self._raise:
            raise LLMError("simulated failure")
        return json.loads(self._reply)


def _src(reliability=0.8) -> SourceRef:
    return SourceRef(type=SourceType.WEB, title="t", snippet="plan A costs $10/mo",
                     reliability=reliability)


def test_supported_claim_gets_blended_confidence() -> None:
    src = _src(reliability=0.8)
    claim = Claim(text="Plan A costs $10", supporting_source_ids=[src.source_id])
    reply = json.dumps({"supported": True, "confidence": 0.9, "contradiction": ""})

    verify_claim(claim, [src], CostTracker(), llm=FakeLLM(reply))

    assert claim.supported is True
    # confidence = model_conf(0.9) * reliability(0.8) = 0.72
    assert abs(claim.confidence - 0.72) < 1e-6
    assert claim.contradiction_note is None
    assert claim.is_grounded


def test_unsupported_claim_gets_zero_confidence() -> None:
    src = _src()
    claim = Claim(text="Plan A is free", supporting_source_ids=[src.source_id])
    reply = json.dumps({"supported": False, "confidence": 0.9,
                        "contradiction": "source says $10/mo, not free"})

    verify_claim(claim, [src], CostTracker(), llm=FakeLLM(reply))

    assert claim.supported is False
    assert claim.confidence == 0.0
    assert claim.contradiction_note == "source says $10/mo, not free"


def test_ungrounded_claim_cannot_be_verified() -> None:
    claim = Claim(text="floating claim", supporting_source_ids=["ghost_id"])
    llm = FakeLLM(json.dumps({"supported": True, "confidence": 1.0}))

    verify_claim(claim, [_src()], CostTracker(), llm=llm)

    assert claim.supported is False
    assert claim.confidence == 0.0
    assert llm.calls == 0, "no cited source resolves -> skip the LLM entirely"


def test_falls_back_to_reliability_heuristic_on_llm_error() -> None:
    src = _src(reliability=0.6)
    claim = Claim(text="Plan A costs $10", supporting_source_ids=[src.source_id])

    verify_claim(claim, [src], CostTracker(), llm=FakeLLM("", raise_error=True))

    assert claim.supported is True, "grounded claim stays supported under fallback"
    assert abs(claim.confidence - 0.6) < 1e-6, "fallback confidence = source reliability"


def test_verify_results_flows_into_report() -> None:
    src = _src(reliability=0.75)
    result = SubTaskResult(
        sub_task_id="task_1",
        status=SubTaskStatus.DONE,
        claims=[Claim(text="Plan A costs $10", supporting_source_ids=[src.source_id])],
        sources=[src],
    )
    reply = json.dumps({"supported": True, "confidence": 0.8, "contradiction": ""})

    verify_results([result], CostTracker(), llm=FakeLLM(reply))

    plan = Plan(question="cost?", sub_tasks=[SubTask(sub_task_id="task_1",
                                                     description="cost of A")])
    report = synthesize(plan, [result])

    # overall = blended confidence 0.8 * 0.75 = 0.6
    assert abs(report.overall_confidence - 0.6) < 1e-6
    assert report.overall_confidence > 0.0, "confidence is no longer stuck at 0%"


def test_contradictions_surface_in_report() -> None:
    src = _src()
    result = SubTaskResult(
        sub_task_id="task_1",
        status=SubTaskStatus.DONE,
        claims=[Claim(text="Plan A is free", supporting_source_ids=[src.source_id])],
        sources=[src],
    )
    reply = json.dumps({"supported": False, "confidence": 0.0,
                        "contradiction": "source says $10/mo"})
    verify_results([result], CostTracker(), llm=FakeLLM(reply))

    plan = Plan(question="cost?", sub_tasks=[SubTask(sub_task_id="task_1",
                                                     description="cost of A")])
    report = synthesize(plan, [result])

    assert len(report.contradictions) == 1
    assert "source says $10/mo" in report.contradictions[0]


if __name__ == "__main__":
    test_supported_claim_gets_blended_confidence()
    test_unsupported_claim_gets_zero_confidence()
    test_ungrounded_claim_cannot_be_verified()
    test_falls_back_to_reliability_heuristic_on_llm_error()
    test_verify_results_flows_into_report()
    test_contradictions_surface_in_report()
    print("VERIFIER TESTS PASSED")
