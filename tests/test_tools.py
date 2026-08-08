"""Tool tests: calculator and documents, planner tool selection, dispatch.

Deterministic: FakeLLM, temp document roots, no network, no API key.
Run: .venv\\Scripts\\python.exe -m pytest tests/test_tools_week7.py -v
"""

from __future__ import annotations

import json

import pytest

from research_agent.core.config import get_settings
from research_agent.core.contracts import SourceType, SubTask
from research_agent.core.llm import LLMError
from research_agent.executor.mcp_servers.calculator import (
    CalcError,
    CalculatorTool,
    safe_eval,
)
from research_agent.executor.mcp_servers.documents import DocumentTool
from research_agent.executor.runner import run_subtask
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
            tracker.record_llm_call(model or "fake", 80, 20)
        if self._raise:
            raise LLMError("simulated failure")
        return json.loads(self._reply)


# --- safe evaluator ----------------------------------------------------------


def test_safe_eval_computes_compound_growth() -> None:
    assert abs(safe_eval("100 * 1.025 ** 3") - 107.6890625) < 1e-9


def test_safe_eval_rejects_code() -> None:
    for evil in (
        "__import__('os').system('dir')",
        "open('x')",
        "a + 1",
        "'x' + 'y'",
        "[1, 2][0]",
    ):
        with pytest.raises(CalcError):
            safe_eval(evil)


def test_safe_eval_rejects_huge_exponent() -> None:
    with pytest.raises(CalcError):
        safe_eval("9 ** 9 ** 9")


def test_safe_eval_rejects_division_by_zero() -> None:
    with pytest.raises(CalcError):
        safe_eval("1 / 0")


# --- calculator tool -----------------------------------------------------------


def test_calculator_uses_llm_extraction_and_exact_math() -> None:
    tool = CalculatorTool(
        llm=FakeLLM('{"expression": "100 * 1.025 ** 3"}'), tracker=CostTracker()
    )
    sources = tool.search("Lãi kép 2.5%/năm của 100 triệu trong 3 năm là bao nhiêu?")
    assert len(sources) == 1
    src = sources[0]
    assert src.type == SourceType.CALCULATOR
    assert src.reliability == 1.0, "exact arithmetic carries no doubt"
    assert "107.6890625" in src.snippet


def test_calculator_falls_back_to_regex_without_llm() -> None:
    tool = CalculatorTool(llm=FakeLLM("", raise_error=True))
    sources = tool.search("What is 2 ** 10 in bytes?")
    assert sources and "1024" in sources[0].snippet


def test_calculator_returns_empty_when_nothing_to_compute() -> None:
    tool = CalculatorTool(llm=FakeLLM('{"expression": ""}'))
    assert tool.search("Thủ đô của Úc là gì?") == []


def test_calculator_rejects_unsafe_llm_expression() -> None:
    tool = CalculatorTool(llm=FakeLLM('{"expression": "__import__(\'os\')"}'))
    assert tool.search("anything") == [], "unsafe expression -> no source, no crash"


# --- document tool --------------------------------------------------------------


def _docs_root(tmp_path):
    (tmp_path / "goi_bao_hiem_a.md").write_text(
        "# Gói bảo hiểm A\n\nPhí hằng năm: 5 triệu.\nQuyền lợi: nội trú 100 triệu.",
        encoding="utf-8",
    )
    (tmp_path / "database_notes.txt").write_text(
        "Postgres vs MongoDB comparison for SaaS workloads.", encoding="utf-8"
    )
    return tmp_path


def test_document_tool_returns_matching_internal_source(tmp_path) -> None:
    tool = DocumentTool(root=_docs_root(tmp_path))
    sources = tool.search("phí gói bảo hiểm a bao nhiêu")
    assert sources
    src = sources[0]
    assert src.type == SourceType.INTERNAL_RAG
    assert src.title == "goi_bao_hiem_a.md"
    assert src.url == "goi_bao_hiem_a.md"
    assert "5 triệu" in src.snippet
    assert src.reliability and src.reliability < 1.0


def test_document_tool_empty_when_root_missing(tmp_path) -> None:
    assert DocumentTool(root=tmp_path / "nope").search("anything") == []


def test_document_tool_never_leaves_its_root(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("bí mật về phí", encoding="utf-8")
    root = tmp_path / "docs"
    root.mkdir()
    assert DocumentTool(root=root).search("phí bí mật") == []


# --- planner tool selection -------------------------------------------------------


def test_planner_maps_tool_hints() -> None:
    reply = json.dumps({"sub_tasks": [
        {"id": "t1", "description": "compute growth", "depends_on": [], "tool": "calculator"},
        {"id": "t2", "description": "read policy doc", "depends_on": [], "tool": "documents"},
        {"id": "t3", "description": "market price", "depends_on": [], "tool": "web"},
        {"id": "t4", "description": "legacy item without tool field", "depends_on": []},
        {"id": "t5", "description": "unknown tool name", "depends_on": [], "tool": "magic"},
    ]})
    plan = plan_question("q", tracker=CostTracker(), llm=FakeLLM(reply))
    hints = [t.tool_hint for t in plan.sub_tasks]
    assert hints == [
        SourceType.CALCULATOR,
        SourceType.INTERNAL_RAG,
        SourceType.WEB,
        SourceType.WEB,  # missing tool -> web
        SourceType.WEB,  # unknown tool -> web
    ]


# --- executor dispatch --------------------------------------------------------------


def test_runner_dispatches_calculator_hint() -> None:
    task = SubTask(description="What is 2 ** 8?", tool_hint=SourceType.CALCULATOR)
    tracker = CostTracker()
    result = run_subtask(
        task, tracker, search_tool=StubSearchTool(),
        llm=FakeLLM('{"expression": "2 ** 8"}'),
    )
    assert result.status.value == "done"
    assert result.sources[0].type == SourceType.CALCULATOR
    assert "256" in result.sources[0].snippet
    assert tracker.snapshot().tool_calls == 1, "calculator counts as a tool call"
    assert result.claims, "claims are distilled from the calculation too"


def test_runner_dispatches_documents_hint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "document_root", str(_docs_root(tmp_path)))
    task = SubTask(
        description="Phí gói bảo hiểm A là bao nhiêu?", tool_hint=SourceType.INTERNAL_RAG
    )
    result = run_subtask(task, CostTracker(), search_tool=StubSearchTool())
    assert result.status.value == "done"
    assert result.sources and result.sources[0].type == SourceType.INTERNAL_RAG


def test_runner_keeps_web_for_plain_hint() -> None:
    task = SubTask(description="anything", tool_hint=SourceType.WEB)
    result = run_subtask(task, CostTracker(), search_tool=StubSearchTool())
    assert result.sources and result.sources[0].type == SourceType.WEB
