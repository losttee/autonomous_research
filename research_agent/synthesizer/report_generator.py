"""Synthesizer: assemble sub-task results into a FinalReport with citations.

synthesize_llm() writes the prose; a citation check strips any [id] the model
invented. Confidence bands and overall confidence come from the verifier's
scores, never from the model. synthesize() is the template fallback used on
any LLM/JSON error.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from research_agent.core.contracts import (
    ConfidenceBand,
    FinalReport,
    Plan,
    ReportSection,
    SourceRef,
    SubTaskResult,
    SubTaskStatus,
)
from research_agent.core.llm import LLMClient, LLMError, get_llm_client
from research_agent.core.logging import get_logger, log_step

_log = get_logger("synthesizer")

_CITATION_RE = re.compile(r"\[(src_[0-9a-f]+|mem_[0-9a-f]+)\]")


def _band(confidence: float) -> ConfidenceBand:
    if confidence >= 0.66:
        return ConfidenceBand.HIGH
    if confidence >= 0.33:
        return ConfidenceBand.MEDIUM
    if confidence > 0.0:
        return ConfidenceBand.LOW
    return ConfidenceBand.UNKNOWN


def _section_for(task_desc: str, result: SubTaskResult) -> ReportSection:
    """Build one report section from a sub-task result, with inline [source_id]."""
    lines: list[str] = []
    cited: list[str] = []
    for claim in result.claims:
        tags = "".join(f"[{sid}]" for sid in claim.supporting_source_ids)
        lines.append(f"{claim.text} {tags}".strip())
        cited.extend(claim.supporting_source_ids)
    body = "\n".join(lines) if lines else "(no supporting evidence found)"
    return ReportSection(
        heading=task_desc,
        body=body,
        cited_source_ids=list(dict.fromkeys(cited)),  # dedupe, keep order
        confidence_band=_band(result.mean_confidence),
    )


def synthesize(
    plan: Plan,
    results: list[SubTaskResult],
) -> FinalReport:
    """Merge sub-task results into a FinalReport. Records uncertainties transparently."""
    desc_by_id = {t.sub_task_id: t.description for t in plan.sub_tasks}

    sections: list[ReportSection] = []
    all_sources: list[SourceRef] = []
    uncertainties: list[str] = []
    contradictions: list[str] = []
    seen_sources: set[str] = set()
    confidences: list[float] = []

    for result in results:
        desc = desc_by_id.get(result.sub_task_id, result.sub_task_id)
        if result.status in (SubTaskStatus.FAILED, SubTaskStatus.SKIPPED):
            uncertainties.append(f"'{desc}': {result.error or result.status.value}")
            continue
        sections.append(_section_for(desc, result))
        confidences.append(result.mean_confidence)
        for claim in result.claims:
            if claim.contradiction_note:
                contradictions.append(f"'{desc}': {claim.contradiction_note}")
        for src in result.sources:
            if src.source_id not in seen_sources:
                seen_sources.add(src.source_id)
                all_sources.append(src)

    overall = sum(confidences) / len(confidences) if confidences else 0.0

    if sections:
        recommendation = (
            f"Synthesized {len(sections)} of {len(results)} sub-tasks from "
            f"{len(all_sources)} sources. See sections for details."
        )
    else:
        recommendation = "No sub-task produced usable evidence; cannot recommend."

    report = FinalReport(
        question=plan.question,
        plan_id=plan.plan_id,
        recommendation=recommendation,
        sections=sections,
        all_sources=all_sources,
        overall_confidence=overall,
        uncertainties=uncertainties,
        contradictions=contradictions,
    )

    log_step(
        _log,
        step_type="synthesize",
        step_id=report.report_id,
        msg="report assembled",
        extra={
            "sections": len(sections),
            "sources": len(all_sources),
            "uncertainties": len(uncertainties),
            "contradictions": len(contradictions),
            "overall_confidence": round(overall, 3),
        },
    )
    return report


# --- LLM synthesis -------------------------------------------------

_SYNTH_SYSTEM = (
    "You are a research synthesizer. Write a clear, well-organized report that "
    "answers the question using ONLY the verified claims provided. Every sentence "
    "that states a fact must carry the [source_id] citation(s) of the claim(s) it "
    "comes from, copied verbatim. Do not invent facts or citation ids. Do not "
    "state confidence numbers. Return STRICT JSON only."
)

_SYNTH_TEMPLATE = """Question:
{question}

Verified claims (cite these source ids verbatim, e.g. [src_ab12]):
{claims}

Return a JSON object of this exact shape:
{{
  "recommendation": "<1-3 sentence direct answer / recommendation, with [source_id] citations>",
  "sections": [
    {{"heading": "<short heading>", "body": "<prose with inline [source_id] citations>"}}
  ]
}}

Rules:
- Use ONLY the claims above. Every factual sentence cites the [source_id](s) it rests on.
- Copy source ids exactly as given; never invent an id.
- 1 to 5 sections. Output JSON only, no prose outside the JSON."""


def _verified_claims_block(results: list[SubTaskResult]) -> tuple[str, set[str]]:
    """Render supported claims as 'text [id][id]' lines; return (block, valid_ids)."""
    lines: list[str] = []
    valid_ids: set[str] = set()
    for result in results:
        if result.status != SubTaskStatus.DONE:
            continue
        for claim in result.claims:
            if not claim.supported:
                continue
            tags = "".join(f"[{sid}]" for sid in claim.supporting_source_ids)
            lines.append(f"{claim.text} {tags}".strip())
            valid_ids.update(claim.supporting_source_ids)
    return "\n".join(lines), valid_ids


def _strip_invalid_citations(text: str, valid_ids: set[str]) -> tuple[str, int]:
    """Remove any [id] the model wrote that isn't a real source. Returns
    (clean_text, dropped_count)."""
    dropped = 0

    def _sub(match: "re.Match[str]") -> str:
        nonlocal dropped
        if match.group(1) in valid_ids:
            return match.group(0)
        dropped += 1
        return ""

    clean = _CITATION_RE.sub(_sub, text)
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    return clean, dropped


def _sections_from_json(data: Any, valid_ids: set[str]) -> tuple[list[dict], int]:
    """Parse the model's sections, stripping invalid citations. Raises ValueError
    on an unusable shape so the caller can fall back to the template."""
    if not isinstance(data, dict):
        raise ValueError("synthesis JSON is not an object")
    raw = data.get("sections")
    if not isinstance(raw, list) or not raw:
        raise ValueError("synthesis JSON has no sections list")

    total_dropped = 0
    parsed: list[dict] = []
    for item in raw[:5]:
        if not isinstance(item, dict):
            continue
        heading = str(item.get("heading", "")).strip() or "Section"
        body_raw = str(item.get("body", "")).strip()
        if not body_raw:
            continue
        body, dropped = _strip_invalid_citations(body_raw, valid_ids)
        total_dropped += dropped
        cited = [sid for sid in _CITATION_RE.findall(body) if sid in valid_ids]
        parsed.append(
            {"heading": heading, "body": body, "cited": list(dict.fromkeys(cited))}
        )
    if not parsed:
        raise ValueError("synthesis produced no usable sections")
    return parsed, total_dropped


def _collect_transparency(
    plan: Plan, results: list[SubTaskResult]
) -> tuple[list[SourceRef], list[str], list[str], float]:
    """Deterministically gather sources, uncertainties, contradictions, and the
    overall confidence from verifier scores."""
    desc_by_id = {t.sub_task_id: t.description for t in plan.sub_tasks}
    all_sources: list[SourceRef] = []
    seen: set[str] = set()
    uncertainties: list[str] = []
    contradictions: list[str] = []
    confidences: list[float] = []

    for result in results:
        desc = desc_by_id.get(result.sub_task_id, result.sub_task_id)
        if result.status in (SubTaskStatus.FAILED, SubTaskStatus.SKIPPED):
            uncertainties.append(f"'{desc}': {result.error or result.status.value}")
            continue
        confidences.append(result.mean_confidence)
        for claim in result.claims:
            if claim.contradiction_note:
                contradictions.append(f"'{desc}': {claim.contradiction_note}")
        for src in result.sources:
            if src.source_id not in seen:
                seen.add(src.source_id)
                all_sources.append(src)

    overall = sum(confidences) / len(confidences) if confidences else 0.0
    return all_sources, uncertainties, contradictions, overall


def synthesize_llm(
    plan: Plan,
    results: list[SubTaskResult],
    tracker: Optional[Any] = None,
    llm: Optional[LLMClient] = None,
) -> FinalReport:
    """LLM-written report over verified claims; falls back to the template on error.

    The LLM only writes prose + citations. Confidence, sources, uncertainties, and
    contradictions are computed deterministically from the verifier so numbers and
    grounding can't be hallucinated. Any invalid [id] the model emits is stripped.
    """
    from research_agent.core.config import get_settings

    claims_block, valid_ids = _verified_claims_block(results)
    all_sources, uncertainties, contradictions, overall = _collect_transparency(
        plan, results
    )

    # No grounded evidence -> nothing for the LLM to write; use the template path.
    if not claims_block or not valid_ids:
        return synthesize(plan, results)

    client = llm or get_llm_client()
    model = get_settings().synth_model
    try:
        data = client.complete_json(
            _SYNTH_TEMPLATE.format(question=plan.question, claims=claims_block),
            model=model,
            system=_SYNTH_SYSTEM,
            max_tokens=1200,
            tracker=tracker,
        )
        parsed, dropped = _sections_from_json(data, valid_ids)
        rec_raw = str(data.get("recommendation", "")).strip()
        recommendation, rec_dropped = _strip_invalid_citations(rec_raw, valid_ids)
        dropped += rec_dropped
        if not recommendation:
            recommendation = f"Synthesized {len(parsed)} sections from {len(all_sources)} sources."
    except (LLMError, ValueError) as exc:
        _log.warning(
            "LLM synthesis fell back to template",
            extra={"extra_fields": {"error": str(exc)}},
        )
        return synthesize(plan, results)

    # Bands use overall confidence; the LLM prose no longer maps per sub-task.
    band = _band(overall)
    sections = [
        ReportSection(
            heading=sec["heading"],
            body=sec["body"],
            cited_source_ids=sec["cited"],
            confidence_band=band,
        )
        for sec in parsed
    ]

    report = FinalReport(
        question=plan.question,
        plan_id=plan.plan_id,
        recommendation=recommendation,
        sections=sections,
        all_sources=all_sources,
        overall_confidence=overall,
        uncertainties=uncertainties,
        contradictions=contradictions,
    )
    log_step(
        _log,
        step_type="synthesize_llm",
        step_id=report.report_id,
        msg="LLM report assembled",
        extra={
            "sections": len(sections),
            "sources": len(all_sources),
            "dropped_citations": dropped,
            "overall_confidence": round(overall, 3),
        },
    )
    return report
