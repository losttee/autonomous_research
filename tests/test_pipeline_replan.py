"""Pipeline replan tests: the MAX_REPLAN loop and budget boundaries.

Run: .venv/Scripts/python.exe -m pytest tests/test_pipeline_replan.py -v
"""

from __future__ import annotations

import json

from research_agent.core.config import get_settings
from research_agent.core.pipeline import run_research
from research_agent.guardrail.cost_tracker import CostTracker


class FakeLLM:
    """Stand-in LLMClient: returns a canned planner JSON and records fake cost."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    def complete_json(self, prompt, *, model=None, system=None, max_tokens=1024,
                      temperature=0.2, tracker=None):
        self.calls += 1
        if tracker is not None:
            tracker.record_llm_call(model or "fake", 100, 50)
        return json.loads(self._reply)


class EmptySearchTool:
    """Every search returns nothing -> every sub-task fails -> evidence is thin."""

    name = "empty"

    def search(self, query: str, max_results: int = 5):
        return []


_PLAN_REPLY = json.dumps(
    {"sub_tasks": [{"id": "t1", "description": "q", "depends_on": []}]}
)


def test_replan_loop_runs_up_to_max_replan() -> None:
    llm = FakeLLM(_PLAN_REPLY)
    outcome = run_research(
        "q", search_tool=EmptySearchTool(), use_memory=False,
        llm=llm, max_replan=2, return_details=True,
    )
    # initial plan + 2 re-plans; with zero sources no other layer can call the LLM
    assert llm.calls == 3
    assert outcome.plan.revision == 2
    assert outcome.plan.previous_plan_id, "revisions must chain to their predecessor"
    assert outcome.plan.replan_reason
    assert outcome.report.uncertainties, "a thin run must surface its gaps"


def test_replan_disabled_stops_at_first_plan() -> None:
    llm = FakeLLM(_PLAN_REPLY)
    outcome = run_research(
        "q", search_tool=EmptySearchTool(), use_memory=False,
        llm=llm, max_replan=0, return_details=True,
    )
    assert llm.calls == 1
    assert outcome.plan.revision == 0
    assert outcome.plan.previous_plan_id is None


def test_replan_stops_when_budget_exhausted() -> None:
    llm = FakeLLM(_PLAN_REPLY)
    tracker = CostTracker(settings=get_settings().model_copy())
    tracker.settings.max_llm_calls = 1  # the first plan consumes the only call
    outcome = run_research(
        "q", tracker=tracker, search_tool=EmptySearchTool(), use_memory=False,
        llm=llm, max_replan=3, return_details=True,
    )
    assert llm.calls == 1
    assert outcome.plan.revision == 0, "budget cap must win over the replan cap"


def test_return_details_false_still_returns_bare_report() -> None:
    llm = FakeLLM(_PLAN_REPLY)
    report = run_research(
        "q", search_tool=EmptySearchTool(), use_memory=False,
        llm=llm, max_replan=0,
    )
    assert report.recommendation, "default call shape stays backward compatible"


if __name__ == "__main__":
    test_replan_loop_runs_up_to_max_replan()
    test_replan_disabled_stops_at_first_plan()
    test_replan_stops_when_budget_exhausted()
    test_return_details_false_still_returns_bare_report()
    print("PIPELINE REPLAN TESTS PASSED")
