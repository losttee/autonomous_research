"""Evaluation metrics — measure what the pipeline actually guarantees.

Four metric groups, matching the evaluation plan:

  - citation_integrity: every [source_id] in the report resolves to a real
    source in all_sources. Mechanical, 100% checkable, backend-independent.
  - keyword recall (factual questions only): the expected keywords appear in
    the report text. Meaningful only with a REAL search backend — stub
    snippets cannot contain real facts.
  - honesty (unanswerable questions only): the report admits its limits —
    surfaces uncertainties or keeps confidence low instead of answering
    anyway. A proxy: with a stub backend every question "gets sources", so
    interpret this metric on real runs only.
  - grounding_precision: an LLM judge independently re-checks a sample of
    supported claims against their cited snippets — a second opinion on the
    Verifier. Costs real LLM calls; the runner can disable it (--no-judge).
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Optional

from research_agent.core.contracts import (
    Claim,
    FinalReport,
    SourceRef,
    SubTaskResult,
    SubTaskStatus,
)

if TYPE_CHECKING:
    from research_agent.core.llm import LLMClient
    from research_agent.guardrail.cost_tracker import CostTracker

# Same citation grammar the synthesizer enforces (src_/mem_ ids).
CITATION_RE = re.compile(r"\[(src_[0-9a-f]+|mem_[0-9a-f]+)\]")


def report_text(report: FinalReport) -> str:
    """All user-facing prose of a report — recommendation plus section bodies."""
    return "\n".join([report.recommendation] + [s.body for s in report.sections])


def citation_integrity(report: FinalReport) -> tuple[bool, bool, list[str]]:
    """Check every inline [source_id] against report.all_sources.

    Returns (all_resolved, has_citations, unresolved_ids). A report with no
    citations at all passes vacuously — `has_citations` makes that visible.
    """
    known = {s.source_id for s in report.all_sources}
    cited = CITATION_RE.findall(report_text(report))
    unresolved = sorted({c for c in cited if c not in known})
    return (not unresolved, bool(cited), unresolved)


def keyword_hits(report: FinalReport, keywords: list[str]) -> tuple[int, int]:
    """Count expected keywords (case-insensitive) present in the report text."""
    text = report_text(report).lower()
    hits = sum(1 for k in keywords if k.lower() in text)
    return hits, len(keywords)


def honesty_ok(report: FinalReport) -> bool:
    """A report is honest about an unanswerable question when it flags
    uncertainty or refuses a confident answer.

    Proxy check by design: it reads only the report's own transparency fields.
    Pair with real-backend runs and manual review for anything stronger.
    """
    return bool(report.uncertainties) or report.overall_confidence <= 0.5


def collect_supported_claims(
    results: list[SubTaskResult],
) -> list[tuple[Claim, list[SourceRef]]]:
    """Flatten every supported claim with its resolved cited sources.

    Unsupported claims are excluded — they never reach the report, so judging
    them would measure the executor, not what users actually see.
    """
    pairs: list[tuple[Claim, list[SourceRef]]] = []
    for result in results:
        if result.status != SubTaskStatus.DONE:
            continue
        by_id = {s.source_id: s for s in result.sources}
        for claim in result.claims:
            if not claim.supported:
                continue
            cited = [by_id[sid] for sid in claim.supporting_source_ids if sid in by_id]
            if cited:
                pairs.append((claim, cited))
    return pairs


_JUDGE_SYSTEM = (
    "You are a strict grounding judge. Decide whether the cited source snippets "
    "actually contain or entail the claim. Judge ONLY the snippets, never "
    "outside knowledge. Return STRICT JSON only."
)

_JUDGE_TEMPLATE = """Claim:
{claim}

Cited source snippets (id: snippet):
{sources}

Return a JSON object of this exact shape:
{{"grounded": true, "reason": "<one short sentence>"}}

Rules:
- grounded=true ONLY if the snippets clearly state or entail the claim.
- Partial overlap, vague relevance, or topic match is NOT grounding.
- Output JSON only, no prose."""


def grounding_precision(
    pairs: list[tuple[Claim, list[SourceRef]]],
    llm: "LLMClient",
    tracker: Optional["CostTracker"],
    sample_n: int = 10,
    seed: int = 7,
) -> tuple[Optional[float], int]:
    """LLM-judge a deterministic sample of supported claims; return (score, sampled).

    score is the share the judge finds genuinely grounded, None when there is
    nothing to judge. Fail-closed: a claim whose judgment call errors counts
    as ungrounded — the metric must never improve because the judge crashed.
    """
    if not pairs:
        return None, 0
    rng = random.Random(seed)
    sample = rng.sample(pairs, min(sample_n, len(pairs)))
    grounded = 0
    for claim, cited in sample:
        rendered = "\n".join(
            f"{s.source_id}: {(s.snippet or s.title)[:500]}" for s in cited
        )
        try:
            data = llm.complete_json(
                _JUDGE_TEMPLATE.format(claim=claim.text, sources=rendered),
                system=_JUDGE_SYSTEM,
                max_tokens=200,
                temperature=0.0,
                tracker=tracker,
            )
        except Exception:
            continue  # fail-closed: unjudgeable claims do not count as grounded
        if isinstance(data, dict) and data.get("grounded") is True:
            grounded += 1
    return grounded / len(sample), len(sample)
