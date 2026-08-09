"""Planner: decompose a question into a Plan of parallelizable sub-tasks.

If the LLM errors or returns malformed output, fall back to a trivial
single-sub-task plan so the pipeline still runs.
"""

from __future__ import annotations

from typing import Any, Optional

from research_agent.core.contracts import Plan, SourceType, SubTask
from research_agent.core.llm import LLMClient, LLMError, get_llm_client
from research_agent.core.logging import get_logger, log_step
from research_agent.guardrail.cost_tracker import CostTracker

_log = get_logger("planner")

_SYSTEM = (
    "You are a research planner. Break a research question into a small set of "
    "independent, concrete sub-tasks. Available tools: 'web' (internet search), "
    "'calculator' (exact arithmetic; use it whenever numbers must be "
    "computed), and 'documents' (internal files / knowledge base). Assign one "
    "tool per sub-task. Prefer 2-5 sub-tasks. Only add a dependency when a "
    "sub-task genuinely needs an earlier one's result. Return STRICT JSON only."
)

_PROMPT_TEMPLATE = """Research question:
{question}

Return a JSON object of this exact shape:
{{
  "sub_tasks": [
    {{"id": "t1", "description": "<one concrete sub-question>", "depends_on": [], "tool": "web"}},
    {{"id": "t2", "description": "...", "depends_on": ["t1"], "tool": "calculator"}}
  ]
}}

Rules:
- 2 to 5 sub-tasks. Each description is self-contained and answerable on its own.
- tool is one of: "web", "calculator", "documents" (default "web").
- depends_on lists ids of sub-tasks that must finish first; use [] when independent.
- Keep independent sub-tasks independent so they can run in parallel.
- Output JSON only, no prose."""

_MAX_SUB_TASKS = 8

# Planner tool names -> source types the executor dispatches on. Unknown or
# missing tool names degrade to web search (the universal fallback).
_TOOL_HINTS: dict[str, SourceType] = {
    "web": SourceType.WEB,
    "calculator": SourceType.CALCULATOR,
    "documents": SourceType.INTERNAL_RAG,
}


def _trivial_plan(question: str) -> Plan:
    """Fallback plan: one sub-task equal to the question."""
    return Plan(question=question, sub_tasks=[SubTask(description=question)])


def _build_plan_from_json(question: str, data: Any) -> Plan:
    """Turn the model's JSON into a Plan, remapping model ids -> real sub_task_ids.

    Raises ValueError if the shape is unusable so the caller can fall back.
    """
    if not isinstance(data, dict):
        raise ValueError("planner JSON is not an object")
    raw_tasks = data.get("sub_tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("planner JSON has no sub_tasks list")

    # First pass: create SubTasks and remember the model's temporary id per task.
    sub_tasks: list[SubTask] = []
    model_ids: list[Optional[str]] = []
    for item in raw_tasks[:_MAX_SUB_TASKS]:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description", "")).strip()
        if not desc:
            continue
        raw_tool = str(item.get("tool", "web")).strip().lower()
        sub_tasks.append(
            SubTask(description=desc, tool_hint=_TOOL_HINTS.get(raw_tool, SourceType.WEB))
        )
        model_ids.append(item.get("id"))

    if not sub_tasks:
        raise ValueError("planner produced no usable sub-tasks")

    # Second pass: translate depends_on (model ids) to real sub_task_ids, dropping
    # any dependency that doesn't resolve to avoid dangling/cyclic references.
    id_map = {mid: st.sub_task_id for mid, st in zip(model_ids, sub_tasks) if mid}
    for item, st in zip(raw_tasks[: len(sub_tasks)], sub_tasks):
        if not isinstance(item, dict):
            continue
        deps = item.get("depends_on") or []
        if isinstance(deps, list):
            st.depends_on = [
                id_map[d]
                for d in deps
                if d in id_map and id_map[d] != st.sub_task_id
            ]

    return Plan(question=question, sub_tasks=sub_tasks)


def plan_question(
    question: str,
    tracker: Optional[CostTracker] = None,
    llm: Optional[LLMClient] = None,
) -> Plan:
    """Decompose the question with the planner LLM; fall back to a trivial plan.

    tracker records the planning LLM call's cost. On any LLM/parse failure the
    trivial single-sub-task plan is returned so the pipeline still runs.
    """
    from research_agent.core.config import get_settings

    client = llm or get_llm_client()
    model = get_settings().planner_model

    try:
        data = client.complete_json(
            _PROMPT_TEMPLATE.format(question=question),
            model=model,
            system=_SYSTEM,
            max_tokens=800,
            tracker=tracker,
        )
        plan = _build_plan_from_json(question, data)
    except (LLMError, ValueError) as exc:
        _log.warning(
            "planner fell back to trivial plan",
            extra={"extra_fields": {"error": str(exc)}},
        )
        plan = _trivial_plan(question)

    batches = plan.parallelizable_batches()
    log_step(
        _log,
        step_type="plan",
        step_id=plan.plan_id,
        msg="plan created",
        extra={
            "sub_tasks": len(plan.sub_tasks),
            "parallel_batches": len(batches),
            "model": model,
        },
    )
    return plan
