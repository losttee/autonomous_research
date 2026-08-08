"""Hardening tests: error injection across every degradation path.

No matter what breaks (LLM outage, garbage output, crashing tools, zero or
mid-run budget, corrupt memory), the pipeline must still return a valid,
honestly-labeled report. Deterministic, no network, no API key.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_hardening.py -v
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from research_agent.core.config import get_settings
from research_agent.core.contracts import (
    Claim,
    FinalReport,
    SourceRef,
    SourceType,
    SubTaskResult,
    SubTaskStatus,
)
from research_agent.core.embeddings import _hash_embedding
from research_agent.core.llm import LLMError
from research_agent.core.pipeline import run_research
from research_agent.executor.web_search import StubSearchTool
from research_agent.guardrail.cost_tracker import CostTracker
from research_agent.memory.store import MemoryStore
from research_agent.memory.vector_store import VectorStore
from research_agent.verifier.verifier import adversarial_pass

from evaluation.metrics import citation_integrity


class FakeEmbedder:
    def embed(self, texts, tracker=None):
        return [_hash_embedding(t) for t in texts]

    def embed_one(self, text, tracker=None):
        return _hash_embedding(text)


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def complete_json(self, prompt, *, model=None, system=None, max_tokens=1024,
                      temperature=0.2, tracker=None):
        if tracker is not None:
            tracker.record_llm_call(model or "fake", 100, 50)
        return json.loads(self._reply)


class GarbageLLM:
    """Valid JSON that no layer can use; forces every parse fallback."""

    def complete_json(self, prompt, **kwargs):
        tracker = kwargs.get("tracker")
        if tracker is not None:
            tracker.record_llm_call("fake", 100, 50)
        return {"junk": 1}


class ExplodingLLM:
    """Every call fails; forces every transport-error fallback."""

    def complete_json(self, prompt, **kwargs):
        raise LLMError("simulated total outage")


class CrashingSearchTool:
    name = "crash"

    def search(self, query: str, max_results: int = 5):
        raise RuntimeError("search backend exploded")


_PLAN_REPLY = json.dumps(
    {"sub_tasks": [{"id": "t1", "description": "look things up", "depends_on": []}]}
)

_DEPENDENT_PLAN = json.dumps({"sub_tasks": [
    {"id": "t1", "description": "first", "depends_on": []},
    {"id": "t2", "description": "second", "depends_on": ["t1"]},
]})


def assert_valid_report(report: FinalReport) -> None:
    """A report is valid when it answers, and every citation resolves."""
    assert report.recommendation
    resolved, _, unresolved = citation_integrity(report)
    assert resolved, f"unresolved citations: {unresolved}"


def test_pipeline_survives_garbage_llm_everywhere() -> None:
    report = run_research(
        "q", search_tool=StubSearchTool(), llm=GarbageLLM(),
        use_memory=False, max_replan=1,
    )
    assert_valid_report(report)
    assert report.sections, "fallback layers still assemble sections"


def test_pipeline_survives_total_llm_outage() -> None:
    report = run_research(
        "q", search_tool=StubSearchTool(), llm=ExplodingLLM(),
        use_memory=False, max_replan=1,
    )
    assert_valid_report(report)
    assert report.sections


def test_crashing_search_surfaces_as_uncertainty() -> None:
    report = run_research(
        "q", search_tool=CrashingSearchTool(), llm=FakeLLM(_PLAN_REPLY),
        use_memory=False, max_replan=0,
    )
    assert_valid_report(report)
    assert report.uncertainties, "a failed sub-task must be admitted, not hidden"
    assert not report.sections


def test_zero_budget_returns_partial_report_not_a_crash() -> None:
    tracker = CostTracker(settings=get_settings().model_copy())
    tracker.settings.max_tool_calls = 0
    report = run_research(
        "q", tracker=tracker, search_tool=StubSearchTool(),
        llm=FakeLLM(_PLAN_REPLY), use_memory=False, max_replan=0,
    )
    assert_valid_report(report)
    assert report.uncertainties


def test_budget_cut_mid_run_still_reports_partial_results() -> None:
    tracker = CostTracker(settings=get_settings().model_copy())
    tracker.settings.max_tool_calls = 1  # first sub-task only
    report = run_research(
        "q", tracker=tracker, search_tool=StubSearchTool(),
        llm=FakeLLM(_DEPENDENT_PLAN), use_memory=False, max_replan=0,
    )
    assert_valid_report(report)
    assert report.sections, "the finished sub-task still earns its section"


def test_corrupted_memory_store_never_breaks_a_run(tmp_path, monkeypatch) -> None:
    corrupt = tmp_path / "memory.json"
    corrupt.write_text("{definitely not json", encoding="utf-8")
    store = MemoryStore(embedder=FakeEmbedder(), store=VectorStore(corrupt))
    monkeypatch.setattr("research_agent.memory.store.get_memory_store", lambda: store)

    report = run_research(
        "q", search_tool=StubSearchTool(), llm=FakeLLM(_PLAN_REPLY),
        use_memory=True, max_replan=0,
    )
    assert_valid_report(report)


def test_sse_stream_emits_error_event_when_pipeline_crashes(monkeypatch) -> None:
    from research_agent.api import main as api_main

    def boom(*args, **kwargs):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(api_main, "run_research", boom)
    client = TestClient(api_main.app)
    resp = client.post("/research/stream", json={"question": "anything at all"})

    assert resp.status_code == 200
    assert "event: error" in resp.text
    assert "pipeline exploded" in resp.text
    assert "event: result" not in resp.text


def test_adversarial_pass_stops_cleanly_on_budget() -> None:
    claim = Claim(text="Plan A costs $10", supporting_source_ids=["src_aaa"],
                  supported=True, confidence=0.8)
    result = SubTaskResult(
        sub_task_id="t1", status=SubTaskStatus.DONE, claims=[claim],
        sources=[SourceRef(type=SourceType.WEB, source_id="src_aaa",
                           snippet="plan A costs $10")],
    )
    tracker = CostTracker(settings=get_settings().model_copy())
    tracker.settings.max_llm_calls = 0

    class Unused:
        calls = 0

        def complete_json(self, prompt, **kwargs):
            Unused.calls += 1
            return {}

    refuted = adversarial_pass([result], tracker, llm=Unused())
    assert refuted == 0 and Unused.calls == 0
    assert claim.supported, "a budget cut must leave claims untouched"
