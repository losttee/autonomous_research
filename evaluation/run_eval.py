"""Run the golden set through the REAL pipeline and record metrics.

This spends real budget: every question goes through planner, search,
verifier, and synthesizer with real LLM calls (guardrail caps still apply
per question). The optional grounding judge spends a little more; disable
it with --no-judge.

Run from the repo root:

    .venv\\Scripts\\python.exe -m evaluation.run_eval
    .venv\\Scripts\\python.exe -m evaluation.run_eval --limit 3 --no-judge

Each run writes evaluation/results/<timestamp>.json and prints a summary.
Memory is OFF by default so runs don't recall or pollute the live store;
pass --use-memory to evaluate recall behavior itself.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from research_agent.core.llm import get_llm_client
from research_agent.core.pipeline import run_research
from research_agent.executor.web_search import get_search_tool
from research_agent.guardrail.cost_tracker import CostTracker

from evaluation.metrics import (
    citation_integrity,
    collect_supported_claims,
    grounding_precision,
    honesty_ok,
    keyword_hits,
)

_GOLDEN = Path(__file__).parent / "golden_set.json"
_RESULTS = Path(__file__).parent / "results"


def _load_golden(limit: int | None) -> list[dict]:
    items = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    return items[:limit] if limit is not None else items


def _evaluate_question(item: dict, args: argparse.Namespace) -> dict:
    """Run one golden question end-to-end and compute its metrics."""
    tracker = CostTracker()
    start = time.monotonic()
    outcome = run_research(
        item["question"],
        tracker=tracker,
        use_memory=args.use_memory,
        return_details=True,
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    report = outcome.report

    resolved, has_cites, unresolved = citation_integrity(report)
    metrics: dict = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "report_id": report.report_id,
        "citation_integrity": resolved,
        "has_citations": has_cites,
        "unresolved_citations": unresolved,
        "overall_confidence": round(report.overall_confidence, 3),
        "grounding_precision": None,
        "grounding_sampled": 0,
    }

    if item["type"] == "factual" and item.get("expect_keywords"):
        hits, total = keyword_hits(report, item["expect_keywords"])
        metrics["keyword_hits"], metrics["keyword_total"] = hits, total
    if item["type"] == "unanswerable":
        metrics["honesty_ok"] = honesty_ok(report)

    if not args.no_judge:
        pairs = collect_supported_claims(outcome.results)
        # Separate tracker: judging cost is eval overhead, not the run's cost.
        score, sampled = grounding_precision(
            pairs, get_llm_client(), CostTracker(),
            sample_n=args.judge_sample, seed=args.seed,
        )
        metrics["grounding_precision"] = round(score, 3) if score is not None else None
        metrics["grounding_sampled"] = sampled

    snap = tracker.snapshot()
    metrics["cost"] = {
        "cost_usd": snap.cost_usd,
        "llm_calls": snap.llm_calls,
        "tool_calls": snap.tool_calls,
        "total_tokens": snap.total_tokens,
        "latency_ms": latency_ms,
    }
    return metrics


def _summarize(results: list[dict], backend: str) -> dict:
    integrity_pass = sum(1 for r in results if r["citation_integrity"])
    honesty = [r for r in results if "honesty_ok" in r]
    honesty_pass = sum(1 for r in honesty if r["honesty_ok"])
    factual = [r for r in results if "keyword_hits" in r]
    keyword_full = sum(
        1 for r in factual if r["keyword_hits"] == r["keyword_total"]
    )
    scores = [r["grounding_precision"] for r in results
              if r["grounding_precision"] is not None]
    return {
        "backend": backend,
        "questions": len(results),
        "citation_integrity_pass": integrity_pass,
        "honesty_pass": honesty_pass,
        "honesty_total": len(honesty),
        "keyword_full_recall": keyword_full,
        "keyword_total": len(factual),
        "avg_grounding_precision": (
            round(sum(scores) / len(scores), 3) if scores else None
        ),
        "total_cost_usd": round(sum(r["cost"]["cost_usd"] for r in results), 6),
        "total_latency_ms": sum(r["cost"]["latency_ms"] for r in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden-set evaluation.")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the LLM grounding judge (saves cost)")
    parser.add_argument("--judge-sample", type=int, default=10,
                        help="max claims per question sent to the judge")
    parser.add_argument("--seed", type=int, default=7,
                        help="seed for the claim sampling")
    parser.add_argument("--use-memory", action="store_true",
                        help="enable memory recall/RAG during the run (default off)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON path (default: results/<timestamp>.json)")
    args = parser.parse_args()

    items = _load_golden(args.limit)
    backend = get_search_tool().name
    if backend == "stub":
        print("NOTE: stub search backend — keyword and honesty metrics are "
              "not meaningful without TAVILY_API_KEY.\n")

    results = [_evaluate_question(item, args) for item in items]
    totals = _summarize(results, backend)

    out = args.out or _RESULTS / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"run_at": datetime.now().isoformat(), "totals": totals, "results": results},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print(f"{'id':<4} {'type':<13} {'cite':<5} {'ground':<7} {'conf':<6} "
          f"{'cost':<8} question")
    for r in results:
        cite = "ok" if r["citation_integrity"] else "FAIL"
        ground = ("-" if r["grounding_precision"] is None
                  else f"{r['grounding_precision']:.2f}")
        print(f"{r['id']:<4} {r['type']:<13} {cite:<5} {ground:<7} "
              f"{r['overall_confidence']:<6.2f} ${r['cost']['cost_usd']:<7.4f} "
              f"{r['question'][:48]}")
    print(f"\ncitation integrity: {totals['citation_integrity_pass']}/{totals['questions']}"
          f" | honesty: {totals['honesty_pass']}/{totals['honesty_total']}"
          f" | keyword full recall: {totals['keyword_full_recall']}/{totals['keyword_total']}"
          f" | avg grounding: {totals['avg_grounding_precision']}"
          f" | total cost: ${totals['total_cost_usd']:.4f}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
