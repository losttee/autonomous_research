"""Executor: run a single sub-task through a tool and return a SubTaskResult.

After web search, the WORKER_MODEL distills the retrieved sources
into concrete claims, each grounded to the source_ids it came from. If the
LLM is unavailable or errors, it falls back to a deterministic heuristic (one claim per
source snippet) so the sub-task still returns something usable.

The executor never raises for tool/LLM errors — it encodes them in the result so
the pipeline can return partial results. Real supported/confidence verification
arrives in the Verifier step; here claims carry an initial confidence.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from research_agent.core.contracts import (
    Claim,
    SourceRef,
    SubTask,
    SubTaskResult,
    SubTaskStatus,
)
from research_agent.core.llm import LLMClient, LLMError, get_llm_client
from research_agent.core.logging import get_logger, log_step
from research_agent.executor.web_search import SearchTool, get_search_tool
from research_agent.guardrail.cost_tracker import BudgetExceeded, CostTracker

_log = get_logger("executor.runner")

_EXTRACT_SYSTEM = (
    "You extract factual claims from search results to answer a specific sub-question. "
    "Use ONLY the provided sources. Every claim must cite the source ids it comes from. "
    "If the sources don't answer the sub-question, return an empty claims list. "
    "Return STRICT JSON only."
)

_EXTRACT_TEMPLATE = """Sub-question:
{question}

Sources (id: content):
{sources}

Return JSON of this exact shape:
{{
  "claims": [
    {{"text": "<one factual claim>", "source_ids": ["<id>", ...]}}
  ]
}}

Rules:
- Only use facts present in the sources above. Do not invent anything.
- Each claim cites the source id(s) that support it. Drop claims you cannot ground.
- Return at most 5 claims. Output JSON only, no prose."""


def _fallback_claims(sources: list[SourceRef]) -> list[Claim]:
    """Fallback heuristic: one claim per source snippet, grounded to that source."""
    return [
        Claim(text=src.snippet or src.title, supporting_source_ids=[src.source_id])
        for src in sources
        if (src.snippet or src.title)
    ]


def _claims_from_json(data: Any, valid_ids: set[str]) -> list[Claim]:
    """Parse the extractor JSON into Claims, keeping only citations to real sources."""
    if not isinstance(data, dict):
        raise ValueError("extractor JSON is not an object")
    raw = data.get("claims")
    if not isinstance(raw, list):
        raise ValueError("extractor JSON has no claims list")

    claims: list[Claim] = []
    for item in raw[:5]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        ids = item.get("source_ids") or []
        if not text or not isinstance(ids, list):
            continue
        grounded = [sid for sid in ids if sid in valid_ids]
        if not grounded:  # ungrounded claim -> drop (no source, no assertion)
            continue
        claims.append(Claim(text=text, supporting_source_ids=grounded))
    return claims


def _extract_claims(
    task: SubTask,
    sources: list[SourceRef],
    tracker: CostTracker,
    llm: Optional[LLMClient],
) -> list[Claim]:
    """Distill sources into grounded claims via the worker LLM; fall back on error."""
    if not sources:
        return []
    from research_agent.core.config import get_settings

    client = llm or get_llm_client()
    model = get_settings().worker_model
    rendered = "\n".join(
        f"{s.source_id}: {(s.snippet or s.title)[:500]}" for s in sources
    )
    try:
        data = client.complete_json(
            _EXTRACT_TEMPLATE.format(question=task.description, sources=rendered),
            model=model,
            system=_EXTRACT_SYSTEM,
            max_tokens=700,
            tracker=tracker,
        )
        claims = _claims_from_json(data, {s.source_id for s in sources})
        return claims or _fallback_claims(sources)
    except (LLMError, ValueError) as exc:
        # TODO: consider adding exponential backoff retry for transient gateway 502/504 errors
        _log.warning(
            "claim extraction fell back to snippets",
            extra={"extra_fields": {"sub_task_id": task.sub_task_id, "error": str(exc)}},
        )
        return _fallback_claims(sources)


def _augment_with_memory(
    task: SubTask,
    sources: list[SourceRef],
    tracker: CostTracker,
) -> list[SourceRef]:
    """Append relevant past claims (as MEMORY SourceRefs) to the web sources.

    Disabled via USE_MEMORY. Never raises — memory is best-effort, so any failure
    just leaves the web sources untouched."""
    from research_agent.core.config import get_settings

    if not get_settings().use_memory:
        return sources
    try:
        from research_agent.memory.store import get_memory_store

        recalled = get_memory_store().search_memory(task.description, tracker=tracker)
    except Exception as exc:
        _log.warning(
            "memory augmentation skipped",
            extra={"extra_fields": {"sub_task_id": task.sub_task_id, "error": str(exc)}},
        )
        return sources
    return sources + recalled


def run_subtask(
    task: SubTask,
    tracker: CostTracker,
    search_tool: SearchTool | None = None,
    max_results: int = 5,
    llm: Optional[LLMClient] = None,
) -> SubTaskResult:
    """Execute one sub-task. Never raises for tool errors — encodes them in the result.

    Budget is checked before the tool call; if exceeded, the sub-task is marked
    SKIPPED so the pipeline can still return partial results.
    """
    tool = search_tool or get_search_tool()
    start = time.monotonic()

    try:
        tracker.check()
    except BudgetExceeded as exc:
        return SubTaskResult(
            sub_task_id=task.sub_task_id,
            status=SubTaskStatus.SKIPPED,
            error=f"skipped due to budget: {exc.reason}",
        )

    try:
        sources = tool.search(task.description, max_results=max_results)
        tracker.record_tool_call()
    except Exception as exc:  # defensive: a tool must not crash the whole run
        _log.warning(
            "tool crashed",
            extra={"extra_fields": {"sub_task_id": task.sub_task_id, "error": str(exc)}},
        )
        return SubTaskResult(
            sub_task_id=task.sub_task_id,
            status=SubTaskStatus.FAILED,
            error=str(exc),
        )

    # Internal RAG: past verified claims become extra sources for this sub-task,
    # cited alongside fresh web results. Best-effort — memory must not break a run.
    sources = _augment_with_memory(task, sources, tracker)

    claims = _extract_claims(task, sources, tracker, llm)

    latency_ms = int((time.monotonic() - start) * 1000)
    status = SubTaskStatus.DONE if sources else SubTaskStatus.FAILED
    result = SubTaskResult(
        sub_task_id=task.sub_task_id,
        status=status,
        claims=claims,
        sources=sources,
        error=None if sources else "no sources found",
        latency_ms=latency_ms,
    )

    log_step(
        _log,
        step_type="subtask",
        step_id=task.sub_task_id,
        latency_ms=latency_ms,
        msg=f"executed via {tool.name}",
        extra={"sources": len(sources), "claims": len(claims), "status": status.value},
    )
    return result
