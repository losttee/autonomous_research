"""End-to-end integration tests without network dependencies (uses stub search).

Run: .venv/Scripts/python.exe -m pytest tests/test_pipeline_e2e.py -v
Or as a plain script: .venv/Scripts/python.exe tests/test_pipeline_e2e.py
"""

from __future__ import annotations

import re

from research_agent.core.contracts import SubTask
from research_agent.core.pipeline import run_research
from research_agent.executor.runner import run_subtask
from research_agent.executor.web_search import StubSearchTool
from research_agent.guardrail.cost_tracker import CostTracker


def test_end_to_end_produces_cited_report() -> None:
    report = run_research(
        "Compare insurance plans A and B by premium and benefits",
        search_tool=StubSearchTool(),
        use_memory=False,  # hermetic: don't touch the shared memory store
    )
    assert report.sections, "report must have at least one section"
    assert report.all_sources, "report must carry sources"

    # Every cited [source_id] in the body must resolve to a real source.
    known = {s.source_id for s in report.all_sources}
    cited = set(re.findall(r"\[(src_[0-9a-f]+)\]", "\n".join(s.body for s in report.sections)))
    assert cited, "at least one inline citation expected"
    assert cited <= known, "all inline citations must map to known sources"


def test_budget_cap_stops_gracefully() -> None:
    from research_agent.core.config import get_settings

    # Use an isolated copy so we don't mutate the shared singleton and leak a
    # 0-cap into other tests.
    tracker = CostTracker(settings=get_settings().model_copy())
    tracker.settings.max_tool_calls = 0  # force immediate cap
    result = run_subtask(SubTask(description="anything"), tracker, search_tool=StubSearchTool())
    assert result.status.value == "skipped"
    assert "budget" in (result.error or "")


if __name__ == "__main__":
    test_end_to_end_produces_cited_report()
    test_budget_cap_stops_gracefully()
    print("PIPELINE E2E TESTS PASSED")
