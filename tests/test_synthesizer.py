"""Synthesizer tests: report generation, citation validation, and fallbacks.

Run: .venv/Scripts/python.exe -m pytest tests/test_synthesizer.py -v
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
from research_agent.core.llm import LLMError
from research_agent.synthesizer.report_generator import synthesize_llm


class FakeLLM:
    def __init__(self, reply: str, *, raise_error: bool = False) -> None:
        self._reply = reply
        self._raise = raise_error
        self.calls = 0

    def complete_json(self, prompt, *, model=None, system=None, max_tokens=1024,
                      temperature=0.2, tracker=None):
        self.calls += 1
        if tracker is not None:
            tracker.record_llm_call(model or "fake", 200, 100)
        if self._raise:
            raise LLMError("simulated failure")
        return json.loads(self._reply)


def _plan_and_results():
    src = SourceRef(type=SourceType.WEB, source_id="src_aaa", snippet="Plan A is $10/mo")
    result = SubTaskResult(
        sub_task_id="t1",
        status=SubTaskStatus.DONE,
        claims=[Claim(text="Plan A costs $10/mo", supporting_source_ids=["src_aaa"],
                      supported=True, confidence=0.8)],
        sources=[src],
    )
    plan = Plan(question="How much is plan A?",
                sub_tasks=[SubTask(sub_task_id="t1", description="cost of A")])
    return plan, [result], src


def test_llm_synthesis_keeps_valid_citations() -> None:
    plan, results, src = _plan_and_results()
    reply = json.dumps({
        "recommendation": f"Plan A costs $10/mo [{src.source_id}].",
        "sections": [{"heading": "Cost", "body": f"Plan A is $10 per month [{src.source_id}]."}],
    })
    report = synthesize_llm(plan, results, llm=FakeLLM(reply))
    assert src.source_id in report.recommendation
    assert report.sections[0].cited_source_ids == [src.source_id]
    assert abs(report.overall_confidence - 0.8) < 1e-6, "confidence stays from verifier"


def test_invalid_citations_are_stripped() -> None:
    plan, results, src = _plan_and_results()
    ghost = "src_deadbeef00"  # well-formed id that maps to no real source
    reply = json.dumps({
        "recommendation": f"Plan A costs $10 [{ghost}].",
        "sections": [{"heading": "Cost",
                      "body": f"Real [{src.source_id}] and fake [{ghost}] cite."}],
    })
    report = synthesize_llm(plan, results, llm=FakeLLM(reply))
    body = report.sections[0].body
    assert ghost not in body, "hallucinated id must be stripped"
    assert src.source_id in body, "real id is kept"
    assert ghost not in report.recommendation


def test_falls_back_to_template_on_llm_error() -> None:
    plan, results, src = _plan_and_results()
    report = synthesize_llm(plan, results, llm=FakeLLM("", raise_error=True))
    # Template path still produces a cited section.
    assert report.sections
    assert src.source_id in "\n".join(s.body for s in report.sections)


def test_no_grounded_evidence_uses_template() -> None:
    plan = Plan(question="q", sub_tasks=[SubTask(sub_task_id="t1", description="d")])
    results = [SubTaskResult(sub_task_id="t1", status=SubTaskStatus.FAILED,
                             error="no sources found")]
    llm = FakeLLM(json.dumps({"recommendation": "x", "sections": []}))
    report = synthesize_llm(plan, results, llm=llm)
    assert llm.calls == 0, "no verified claims -> skip the LLM entirely"
    assert report.uncertainties, "failed sub-task surfaces as an uncertainty"


if __name__ == "__main__":
    test_llm_synthesis_keeps_valid_citations()
    test_invalid_citations_are_stripped()
    test_falls_back_to_template_on_llm_error()
    test_no_grounded_evidence_uses_template()
    print("SYNTHESIS TESTS PASSED")
