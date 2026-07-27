"""Verifier: grounding check for extracted claims against cited sources.

The executor produces claims with an unverified confidence of 0.0. This layer is
where each claim earns its confidence: for every claim we ask the cheap
VERIFIER_MODEL whether the cited sources actually entail the claim, and record

  - supported:   True/False  (is the claim entailed by its sources?)
  - confidence:  0.0-1.0      (how strongly, blended with source reliability)
  - contradiction_note: filled when sources disagree with the claim

Degradation is a first-class requirement (same as planner/executor): if the LLM
errors or returns malformed output, fall back to a deterministic heuristic based
on source reliability so the claim still gets a sensible, non-zero confidence.
"""

from __future__ import annotations

from typing import Any, Optional

from research_agent.core.contracts import (
    Claim,
    SourceRef,
    SubTaskResult,
)
from research_agent.core.llm import LLMClient, LLMError, get_llm_client
from research_agent.core.logging import get_logger, log_step
from research_agent.guardrail.cost_tracker import BudgetExceeded, CostTracker

_log = get_logger("verifier")

# Confidence to assume when the LLM verifier is unavailable and we fall back to
# source reliability alone. Reliability in [0,1]; default reliability ~0.5 keeps
# an unscored source at a cautious mid confidence.
_DEFAULT_RELIABILITY = 0.5

_SYSTEM = (
    "You are a fact-checking verifier. Given a claim and the exact source snippets "
    "it cites, decide whether the sources ENTAIL the claim. Judge only from the "
    "provided snippets, not outside knowledge. Return STRICT JSON only."
)

_PROMPT_TEMPLATE = """Claim:
{claim}

Cited sources (id: snippet):
{sources}

Return a JSON object of this exact shape:
{{
  "supported": true,
  "confidence": 0.0,
  "contradiction": "<empty string, or note if a source contradicts the claim>"
}}

Rules:
- supported = true only if the snippets clearly entail the claim.
- confidence in [0.0, 1.0]: how strongly the snippets support the claim.
- If a snippet contradicts the claim, set supported=false and describe it in contradiction.
- Output JSON only, no prose."""


def _sources_by_id(sources: list[SourceRef]) -> dict[str, SourceRef]:
    return {s.source_id: s for s in sources}


def _mean_reliability(claim: Claim, by_id: dict[str, SourceRef]) -> float:
    """Average reliability of the sources this claim cites; default when unscored."""
    scores = [
        (by_id[sid].reliability if by_id[sid].reliability is not None else _DEFAULT_RELIABILITY)
        for sid in claim.supporting_source_ids
        if sid in by_id
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _heuristic_verify(claim: Claim, by_id: dict[str, SourceRef]) -> None:
    """Deterministic fallback: a grounded claim is 'supported' with confidence
    equal to the mean reliability of its sources. Mutates the claim in place."""
    rel = _mean_reliability(claim, by_id)
    claim.supported = bool(claim.supporting_source_ids)
    claim.confidence = round(rel, 3)


def _apply_json(claim: Claim, data: Any, by_id: dict[str, SourceRef]) -> None:
    """Apply the verifier LLM's JSON verdict to the claim, blended with reliability.

    Raises ValueError on an unusable shape so the caller can fall back.
    """
    if not isinstance(data, dict):
        raise ValueError("verifier JSON is not an object")
    if "supported" not in data or "confidence" not in data:
        raise ValueError("verifier JSON missing required fields")

    supported = bool(data.get("supported"))
    try:
        raw_conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"confidence not a number: {exc}") from exc
    conf = min(1.0, max(0.0, raw_conf))

    # Blend the model's judgment with how reliable the underlying sources are, so
    # a confident claim resting on weak sources is discounted.
    rel = _mean_reliability(claim, by_id)
    blended = conf * rel if supported else 0.0

    note = str(data.get("contradiction", "")).strip()
    claim.supported = supported
    claim.confidence = round(blended, 3)
    claim.contradiction_note = note or None


def verify_claim(
    claim: Claim,
    sources: list[SourceRef],
    tracker: CostTracker,
    llm: Optional[LLMClient] = None,
) -> Claim:
    """Verify one claim against its cited sources; mutate + return it.

    Never raises for LLM errors — falls back to the reliability heuristic so the
    claim always ends up with a sensible confidence.
    """
    from research_agent.core.config import get_settings

    by_id = _sources_by_id(sources)
    cited = [by_id[sid] for sid in claim.supporting_source_ids if sid in by_id]
    if not cited:  # ungrounded claim can't be verified -> stays unsupported
        claim.supported = False
        claim.confidence = 0.0
        return claim

    client = llm or get_llm_client()
    model = get_settings().verifier_model
    rendered = "\n".join(f"{s.source_id}: {(s.snippet or s.title)[:500]}" for s in cited)

    try:
        data = client.complete_json(
            _PROMPT_TEMPLATE.format(claim=claim.text, sources=rendered),
            model=model,
            system=_SYSTEM,
            max_tokens=300,
            tracker=tracker,
        )
        _apply_json(claim, data, by_id)
    except (LLMError, ValueError) as exc:
        _log.warning(
            "claim verification fell back to reliability heuristic",
            extra={"extra_fields": {"claim_id": claim.claim_id, "error": str(exc)}},
        )
        _heuristic_verify(claim, by_id)
    return claim


def verify_results(
    results: list[SubTaskResult],
    tracker: CostTracker,
    llm: Optional[LLMClient] = None,
) -> list[SubTaskResult]:
    """Verify every claim across all sub-task results, in place.

    Stops issuing new LLM calls once the budget is exceeded, falling back to the
    heuristic for the remaining claims so verification always completes.
    """
    verified = 0
    supported = 0
    contradictions = 0
    budget_hit = False

    for result in results:
        by_id = _sources_by_id(result.sources)
        for claim in result.claims:
            if budget_hit:
                _heuristic_verify(claim, by_id)
            else:
                try:
                    tracker.check()
                except BudgetExceeded:
                    budget_hit = True
                    _heuristic_verify(claim, by_id)
                else:
                    verify_claim(claim, result.sources, tracker, llm=llm)
            verified += 1
            if claim.supported:
                supported += 1
            if claim.contradiction_note:
                contradictions += 1

    log_step(
        _log,
        step_type="verify",
        step_id="verifier",
        msg="claims verified",
        extra={
            "claims": verified,
            "supported": supported,
            "contradictions": contradictions,
            "budget_hit": budget_hit,
        },
    )
    return results
