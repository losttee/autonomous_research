"""Monitoring & SSE tests — log aggregation, progress events, /metrics, and streaming.

Run: .venv/Scripts/python.exe -m pytest tests/test_monitoring.py -v
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from research_agent.core.config import get_settings
from research_agent.core.pipeline import run_research
from research_agent.executor.web_search import StubSearchTool

from monitoring.aggregate import aggregate


# --- aggregate --------------------------------------------------------------


def _write_logs(tmp_path, lines):
    path = tmp_path / "pipeline.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_aggregate_rollups_and_runs(tmp_path) -> None:
    lines = [
        json.dumps({"step_type": "plan", "tokens": 300, "latency_ms": 800, "cost_usd": 0.003}),
        json.dumps({"step_type": "research_done", "step_id": "rpt_1", "ts": "2026-01-01T00:00:00",
                    "latency_ms": 4000, "cost_usd": 0.02, "sections": 2, "sources": 3}),
        json.dumps({"step_type": "research_done", "step_id": "rpt_2", "ts": "2026-01-02T00:00:00",
                    "latency_ms": 2000, "cost_usd": 0.01, "sections": 1, "sources": 2}),
        json.dumps({"step_type": "budget_exceeded"}),
        "{not valid json — must be skipped",
    ]
    data = aggregate(_write_logs(tmp_path, lines))

    assert data["totals"]["steps"] == 4, "malformed line is skipped"
    assert data["totals"]["budget_exceeded"] == 1
    assert data["run_count"] == 2
    assert data["runs"][0]["report_id"] == "rpt_2", "newest run first"
    assert abs(data["avg_run_cost_usd"] - 0.015) < 1e-9
    steps = {s["step"]: s for s in data["by_step"]}
    assert steps["research_done"]["count"] == 2
    assert steps["plan"]["avg_latency_ms"] == 800


def test_aggregate_missing_file(tmp_path) -> None:
    data = aggregate(tmp_path / "does_not_exist.jsonl")
    assert data["run_count"] == 0
    assert data["totals"]["steps"] == 0
    assert data["by_step"] == [] and data["runs"] == []


# --- pipeline progress events ------------------------------------------------

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


_PLAN_REPLY = json.dumps({
    "sub_tasks": [
        {"id": "t1", "description": "a", "depends_on": []},
        {"id": "t2", "description": "b", "depends_on": ["t1"]},
    ]
})


def test_progress_events_follow_pipeline_order() -> None:
    events: list[str] = []
    run_research(
        "q", search_tool=StubSearchTool(), use_memory=False,
        llm=FakeLLM(_PLAN_REPLY), max_replan=0,
        on_progress=lambda step, msg, **extra: events.append(step),
    )
    assert events[0] == "plan"
    assert "execute" in events and "verify" in events
    assert events[-1] == "synthesize"
    assert events.index("execute") < events.index("verify") < events.index("synthesize")
    assert events.count("execute") == 2, "two dependency batches -> two execute events"


def test_progress_callback_errors_never_break_a_run() -> None:
    def boom(step, msg, **extra):
        raise RuntimeError("listener crashed")

    report = run_research(
        "q", search_tool=StubSearchTool(), use_memory=False,
        llm=FakeLLM(_PLAN_REPLY), max_replan=0, on_progress=boom,
    )
    assert report.sections, "a broken listener must not break the pipeline"


# --- API endpoints -----------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    """TestClient with a fully hermetic pipeline (fake LLM, stub search, no memory)."""
    fake = FakeLLM(_PLAN_REPLY)
    for module in (
        "research_agent.planner.planner",
        "research_agent.executor.runner",
        "research_agent.verifier.verifier",
        "research_agent.synthesizer.report_generator",
    ):
        monkeypatch.setattr(module + ".get_llm_client", lambda: fake)
    monkeypatch.setattr(
        "research_agent.core.pipeline.get_search_tool", lambda: StubSearchTool()
    )
    monkeypatch.setattr(get_settings(), "use_memory", False)

    from research_agent.api import main as api_main

    return TestClient(api_main.app)


def test_metrics_endpoint_aggregates_log_file(client, monkeypatch, tmp_path) -> None:
    log = tmp_path / "m.jsonl"
    log.write_text(
        json.dumps({"step_type": "research_done", "step_id": "rpt_x",
                    "ts": "t", "latency_ms": 1000, "cost_usd": 0.01}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(get_settings(), "log_file", str(log))

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_count"] == 1
    assert body["runs"][0]["report_id"] == "rpt_x"


def test_research_stream_emits_progress_then_result(client) -> None:
    resp = client.post("/research/stream", json={"question": "Compare plans A and B"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    text = resp.text
    assert "event: progress" in text
    assert "event: result" in text
    assert '"report"' in text
    # stages arrive in pipeline order inside the stream
    assert text.index("Planning the research") < text.index("Writing the report")


def test_classic_research_endpoint_still_works(client) -> None:
    resp = client.post("/research", json={"question": "Compare plans A and B"})
    assert resp.status_code == 200
    assert resp.json()["report"]["question"] == "Compare plans A and B"


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    test_aggregate_rollups_and_runs(tmp)
    test_aggregate_missing_file(tmp)
    test_progress_events_follow_pipeline_order()
    test_progress_callback_errors_never_break_a_run()
    print("MONITORING TESTS PASSED")
    sys.exit(0)
