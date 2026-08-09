"""Verifier: check each claim against its cited sources.

For every claim the verifier model decides entailment and sets supported,
confidence (blended with source reliability) and contradiction_note. On LLM
error or malformed output, fall back to a reliability-based heuristic.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from research_agent.core.contracts import (
    Claim,
    SourceRef,
    SubTaskResult,
    SubTaskStatus,
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

    On LLM error, falls back to the reliability heuristic.
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
    adversarial: bool | None = None,
) -> list[SubTaskResult]:
    """Verify every claim across all sub-task results, in place.

    Stops issuing new LLM calls once the budget is exceeded, falling back to the
    heuristic for the remaining claims so verification always completes.
    `adversarial` (default: the ADVERSARIAL_VERIFY setting) adds a refutation
    pass on strong claims (extra cost, off by default).
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

    if adversarial is None:
        from research_agent.core.config import get_settings

        adversarial = get_settings().adversarial_verify
    if adversarial:
        adversarial_pass(results, tracker, llm=llm)
    return results


# --- Consolidation ----------------------------------------------------------
# Merge near-duplicate claims, then flag claims that agree on the topic but
# disagree on the numbers. Both are deterministic (no LLM); flags flow into
# the report through contradiction_note.

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

# Near-identical wording -> same finding.
_DEDUP_JACCARD = 0.8
# Same topic, different numbers -> contradiction candidate.
_CONTRA_TOPIC_JACCARD = 0.3
_MAX_CROSS_FACTS = 100

# Adversarial pass: strong claims only, capped; each check costs an LLM call.
_ADVERSARIAL_MIN_CONFIDENCE = 0.6
_ADVERSARIAL_MAX_CLAIMS = 5

_ADVERSARIAL_SYSTEM = (
    "You are an adversarial fact-checker. Your job is to REFUTE the claim if "
    "the snippets allow it. Judge ONLY the provided snippets, never outside "
    "knowledge. Return STRICT JSON only."
)

_ADVERSARIAL_TEMPLATE = """Claim:
{claim}

Cited source snippets (id: snippet):
{sources}

Return a JSON object of this exact shape:
{{"refuted": true, "reason": "<empty string, or why the snippets contradict the claim>"}}

Rules:
- refuted=true ONLY if the snippets contain direct evidence AGAINST the claim.
- Weakness, vagueness, or missing detail is NOT refutation.
- Output JSON only, no prose."""


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe_claims(results: list[SubTaskResult]) -> list[SubTaskResult]:
    """Merge near-identical claims across all results, keeping the strongest.

    Deterministic token-Jaccard grouping (no LLM). The survivor takes the
    higher confidence and its supported verdict, the union of supporting
    sources, and any contradiction note. Mutates in place and returns.
    """
    kept: list[Claim] = []
    kept_tokens: list[set[str]] = []
    before = 0
    for result in results:
        survivors: list[Claim] = []
        for claim in result.claims:
            before += 1
            claim_tokens = _tokens(claim.text)
            match: Optional[Claim] = None
            for survivor, survivor_tokens in zip(kept, kept_tokens):
                if _jaccard(claim_tokens, survivor_tokens) >= _DEDUP_JACCARD:
                    match = survivor
                    break
            if match is None:
                kept.append(claim)
                kept_tokens.append(claim_tokens)
                survivors.append(claim)
                continue
            merged = list(dict.fromkeys(match.supporting_source_ids + claim.supporting_source_ids))
            match.supporting_source_ids = merged
            if claim.confidence > match.confidence:
                match.confidence = claim.confidence
                match.supported = claim.supported
            if claim.contradiction_note and not match.contradiction_note:
                match.contradiction_note = claim.contradiction_note
        result.claims = survivors
    log_step(
        _log,
        step_type="dedupe",
        step_id="verifier",
        msg="claims deduplicated",
        extra={"claims_before": before, "claims_after": len(kept),
               "merged": before - len(kept)},
    )
    return results


def _numeric_signature(claim: Claim) -> Optional[tuple[set[str], frozenset[str]]]:
    """(topic tokens, numbers) for a claim that carries at least one number."""
    numbers = _NUM_RE.findall(claim.text)
    if not numbers:
        return None
    topics = _tokens(_NUM_RE.sub(" ", claim.text))
    return topics, frozenset(n.replace(",", ".") for n in numbers)


def find_cross_contradictions(
    results: list[SubTaskResult],
) -> list[tuple[Claim, Claim]]:
    """Supported claims about the same topic that give different numbers.

    Deterministic, no LLM. Capped to keep the pairwise scan bounded.
    """
    facts: list[tuple[Claim, set[str], frozenset[str]]] = []
    for result in results:
        if result.status != SubTaskStatus.DONE:
            continue
        for claim in result.claims:
            if not claim.supported:
                continue
            signature = _numeric_signature(claim)
            if signature is not None:
                facts.append((claim, signature[0], signature[1]))

    pairs: list[tuple[Claim, Claim]] = []
    pool = facts[:_MAX_CROSS_FACTS]
    for i, (claim_a, topics_a, nums_a) in enumerate(pool):
        for claim_b, topics_b, nums_b in pool[i + 1:]:
            if nums_a == nums_b:
                continue
            if _jaccard(topics_a, topics_b) >= _CONTRA_TOPIC_JACCARD:
                pairs.append((claim_a, claim_b))
    return pairs


def flag_cross_contradictions(results: list[SubTaskResult]) -> int:
    """Mark conflicting pairs so the synthesis transparency pass surfaces them.

    Both sides get a note quoting the other finding. Returns the pair count.
    """
    pairs = find_cross_contradictions(results)
    for claim_a, claim_b in pairs:
        for target, other in ((claim_a, claim_b), (claim_b, claim_a)):
            note = f"conflicts with another finding: '{other.text}'"
            target.contradiction_note = (
                note if not target.contradiction_note
                else f"{target.contradiction_note}; also {note}"
            )
    if pairs:
        log_step(
            _log,
            step_type="cross_contradictions",
            step_id="verifier",
            msg="conflicting findings across sub-tasks",
            extra={"pairs": len(pairs)},
        )
    return len(pairs)


def adversarial_pass(
    results: list[SubTaskResult],
    tracker: CostTracker,
    llm: Optional[LLMClient] = None,
    max_claims: int = _ADVERSARIAL_MAX_CLAIMS,
) -> int:
    """Second opinion on the strongest claims: ask the model to refute them.

    A refuted claim is retracted: unsupported, zero confidence, reason kept as
    a contradiction note. If the check call fails the claim stands. A budget
    hit ends the pass early. Returns the refuted count.
    """
    from research_agent.core.config import get_settings

    client = llm or get_llm_client()
    model = get_settings().verifier_model
    checked = 0
    refuted = 0

    for result in results:
        if result.status != SubTaskStatus.DONE:
            continue
        by_id = _sources_by_id(result.sources)
        for claim in result.claims:
            if checked >= max_claims:
                log_step(
                    _log, step_type="adversarial", step_id="verifier",
                    msg="adversarial pass capped",
                    extra={"checked": checked, "refuted": refuted},
                )
                return refuted
            if not claim.supported or claim.confidence < _ADVERSARIAL_MIN_CONFIDENCE:
                continue
            cited = [by_id[sid] for sid in claim.supporting_source_ids if sid in by_id]
            if not cited:
                continue
            try:
                tracker.check()
            except BudgetExceeded:
                log_step(
                    _log, step_type="adversarial", step_id="verifier",
                    msg="adversarial pass stopped by budget",
                    extra={"checked": checked, "refuted": refuted},
                )
                return refuted
            rendered = "\n".join(
                f"{s.source_id}: {(s.snippet or s.title)[:500]}" for s in cited
            )
            try:
                data = client.complete_json(
                    _ADVERSARIAL_TEMPLATE.format(claim=claim.text, sources=rendered),
                    model=model,
                    system=_ADVERSARIAL_SYSTEM,
                    max_tokens=300,
                    temperature=0.0,
                    tracker=tracker,
                )
            except LLMError:
                continue  # judge unavailable: claim stands
            checked += 1
            if isinstance(data, dict) and data.get("refuted") is True:
                reason = str(data.get("reason", "")).strip() or "refuted on review"
                claim.supported = False
                claim.confidence = 0.0
                claim.contradiction_note = f"refuted on adversarial review: {reason}"
                refuted += 1

    log_step(
        _log,
        step_type="adversarial",
        step_id="verifier",
        msg="adversarial pass complete",
        extra={"checked": checked, "refuted": refuted},
    )
    return refuted
