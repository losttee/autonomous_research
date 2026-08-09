"""Planner tests: decomposition, parallel execution, and fallback.

Uses a FakeLLM so tests are deterministic and need no API key or network.
Run: .venv/Scripts/python.exe -m pytest tests/test_planner.py -v
"""

from __future__ import annotations

import json

from research_agent.core.contracts import SubTask
from research_agent.core.llm import LLMError
from research_agent.executor.web_search import StubSearchTool
from research_agent.guardrail.cost_tracker import CostTracker
from research_agent.planner.planner import plan_question


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
            tracker.record_llm_call(model or "fake", 100, 50)
        if self._raise:
            raise LLMError("simulated failure")
        return json.loads(self._reply)


def test_planner_decomposes_into_parallel_batches() -> None:
    reply = json.dumps({
        "sub_tasks": [
            {"id": "t1", "description": "Premium of plan A", "depends_on": []},
            {"id": "t2", "description": "Premium of plan B", "depends_on": []},
            {"id": "t3", "description": "Compare A vs B", "depends_on": ["t1", "t2"]},
        ]
    })
    plan = plan_question("Compare A and B", tracker=CostTracker(), llm=FakeLLM(reply))
    assert len(plan.sub_tasks) == 3
    batches = plan.parallelizable_batches()
    assert len(batches) == 2, "t1/t2 run in parallel, t3 depends on them"
    assert len(batches[0]) == 2 and len(batches[1]) == 1


def test_planner_falls_back_on_llm_error() -> None:
    plan = plan_question("anything", tracker=CostTracker(), llm=FakeLLM("", raise_error=True))
    assert len(plan.sub_tasks) == 1, "LLM failure -> trivial single-sub-task plan"
    assert plan.sub_tasks[0].description == "anything"


def test_planner_drops_dangling_dependencies() -> None:
    reply = json.dumps({
        "sub_tasks": [
            {"id": "t1", "description": "task one", "depends_on": ["ghost"]},
        ]
    })
    plan = plan_question("q", tracker=CostTracker(), llm=FakeLLM(reply))
    assert plan.sub_tasks[0].depends_on == [], "unresolved dep ids are dropped"


def test_end_to_end_with_multitask_plan() -> None:
    # run_research builds its own plan; verify the parallel path via _run_batch instead.
    from research_agent.core.pipeline import _run_batch
    tracker = CostTracker()
    tasks = [SubTask(description="Look up premium of plan A"),
             SubTask(description="Look up premium of plan B")]
    results = _run_batch(tasks, tracker, StubSearchTool())
    assert len(results) == 2
    assert all(r.sources for r in results), "each parallel sub-task returns sources"


if __name__ == "__main__":
    test_planner_decomposes_into_parallel_batches()
    test_planner_falls_back_on_llm_error()
    test_planner_drops_dangling_dependencies()
    test_end_to_end_with_multitask_plan()
    print("PLANNER TESTS PASSED")
