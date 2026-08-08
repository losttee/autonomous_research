"""Benchmark harness tests. Deterministic (FakeLLM + stub search), no network.

The benchmark itself spends real budget; these tests only verify the harness
wiring. Run: .venv\\Scripts\\python.exe -m pytest tests/test_benchmark.py -v
"""

from __future__ import annotations

import json

from research_agent.executor.web_search import StubSearchTool

from evaluation.benchmark import compare, run_variant


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def complete_json(self, prompt, *, model=None, system=None, max_tokens=1024,
                      temperature=0.2, tracker=None):
        if tracker is not None:
            tracker.record_llm_call(model or "fake", 100, 50)
        return json.loads(self._reply)


class QueueLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)

    def complete_json(self, prompt, **kwargs):
        tracker = kwargs.get("tracker")
        if tracker is not None:
            tracker.record_llm_call("fake", 50, 20)
        return json.loads(self._replies.pop(0))


_PLAN_REPLY = json.dumps(
    {"sub_tasks": [{"id": "t1", "description": "look things up", "depends_on": []}]}
)

_ITEM = {"id": "f1", "type": "factual", "question": "Compare plans A and B",
         "expect_keywords": ["plan"]}


def test_run_variant_scores_a_single_run() -> None:
    data = run_variant(
        _ITEM["question"], verify_claims=True,
        llm=FakeLLM(_PLAN_REPLY), search_tool=StubSearchTool(),
    )
    assert data["citation_integrity"] is True
    assert data["grounding_precision"] is None, "no judge -> not scored"
    assert data["cost"]["llm_calls"] >= 1
    assert data["cost"]["latency_ms"] >= 0


def test_compare_runs_both_variants() -> None:
    data = compare(_ITEM, llm=FakeLLM(_PLAN_REPLY), search_tool=StubSearchTool())
    assert set(data) == {"id", "question", "type", "with_verifier", "without_verifier"}
    for variant in (data["with_verifier"], data["without_verifier"]):
        assert variant["citation_integrity"] is True
        assert variant["has_citations"] is True


def test_unverified_baseline_passes_claims_straight_through() -> None:
    """Without the verifier, claims keep citations but carry zero confidence;
    that contrast is what the benchmark measures."""
    data = compare(_ITEM, llm=FakeLLM(_PLAN_REPLY), search_tool=StubSearchTool())
    assert data["without_verifier"]["overall_confidence"] == 0.0


def test_judge_scores_both_variants() -> None:
    judge = QueueLLM(['{"grounded": true, "reason": "ok"}'] * 20)
    data = compare(
        _ITEM, llm=FakeLLM(_PLAN_REPLY), search_tool=StubSearchTool(), judge=judge,
    )
    assert data["with_verifier"]["grounding_precision"] == 1.0
    assert data["with_verifier"]["grounding_sampled"] > 0
    assert data["without_verifier"]["grounding_sampled"] > 0
