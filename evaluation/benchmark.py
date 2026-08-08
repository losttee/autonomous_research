"""Benchmark: full pipeline vs. the verifier switched off.

Each golden question runs through both variants (memory off, everything else
identical) and is scored with the run_eval metrics; the delta isolates what
verification buys. Grounding precision uses the LLM judge (--no-judge skips).

COST: two real pipeline runs per question. Use --limit. Results:
evaluation/results/benchmark_<timestamp>.json

    .venv\\Scripts\\python.exe -m evaluation.benchmark --limit 3
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from research_agent.core.llm import get_llm_client
from research_agent.core.pipeline import run_research
from research_agent.executor.web_search import get_search_tool
from research_agent.guardrail.cost_tracker import CostTracker

from evaluation.metrics import (
    citation_integrity,
    collect_supported_claims,
    grounding_precision,
)

_GOLDEN = Path(__file__).parent / "golden_set.json"
_RESULTS = Path(__file__).parent / "results"


def run_variant(
    question: str,
    *,
    verify_claims: bool,
    llm: Any = None,
    search_tool: Any = None,
    judge: Any = None,
    judge_sample: int = 10,
    seed: int = 7,
) -> dict[str, Any]:
    """One pipeline run under a single variant, scored like run_eval."""
    tracker = CostTracker()
    start = time.monotonic()
    outcome = run_research(
        question,
        tracker=tracker,
        use_memory=False,
        verify_claims=verify_claims,
        llm=llm,
        search_tool=search_tool,
        return_details=True,
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    report = outcome.report

    resolved, has_cites, unresolved = citation_integrity(report)
    data: dict[str, Any] = {
        "citation_integrity": resolved,
        "has_citations": has_cites,
        "unresolved_citations": unresolved,
        "overall_confidence": round(report.overall_confidence, 3),
        "grounding_precision": None,
        "grounding_sampled": 0,
    }
    if judge is not None:
        pairs = collect_supported_claims(outcome.results)
        score, sampled = grounding_precision(
            pairs, judge, CostTracker(), sample_n=judge_sample, seed=seed
        )
        data["grounding_precision"] = round(score, 3) if score is not None else None
        data["grounding_sampled"] = sampled

    snap = tracker.snapshot()
    data["cost"] = {
        "cost_usd": snap.cost_usd,
        "llm_calls": snap.llm_calls,
        "tool_calls": snap.tool_calls,
        "total_tokens": snap.total_tokens,
        "latency_ms": latency_ms,
    }
    return data


def compare(item: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Both variants for one golden question, side by side."""
    return {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "with_verifier": run_variant(item["question"], verify_claims=True, **kwargs),
        "without_verifier": run_variant(item["question"], verify_claims=False, **kwargs),
    }


def _avg(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def variant_stats(key: str) -> dict[str, Any]:
        rows = [r[key] for r in results]
        grounding = [r["grounding_precision"] for r in rows
                     if r["grounding_precision"] is not None]
        return {
            "citation_integrity_pass": sum(1 for r in rows if r["citation_integrity"]),
            "avg_grounding_precision": _avg(grounding),
            "avg_confidence": _avg([r["overall_confidence"] for r in rows]),
            "total_cost_usd": round(sum(r["cost"]["cost_usd"] for r in rows), 6),
        }

    with_v = variant_stats("with_verifier")
    without_v = variant_stats("without_verifier")
    delta = None
    if with_v["avg_grounding_precision"] is not None and \
            without_v["avg_grounding_precision"] is not None:
        delta = round(with_v["avg_grounding_precision"]
                      - without_v["avg_grounding_precision"], 3)
    return {
        "questions": len(results),
        "with_verifier": with_v,
        "without_verifier": without_v,
        "grounding_precision_delta": delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifier on/off benchmark.")
    parser.add_argument("--limit", type=int, default=None,
                        help="benchmark only the first N eligible questions")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the LLM grounding judge (saves cost)")
    parser.add_argument("--judge-sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-unanswerable", action="store_true",
                        help="also run the unanswerable questions (default: skip)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    items = [i for i in golden
             if args.include_unanswerable or i["type"] != "unanswerable"]
    if args.limit is not None:
        items = items[: args.limit]

    backend = get_search_tool().name
    if backend == "stub":
        print("NOTE: stub search backend; grounding numbers reflect stub "
              "snippets, not real evidence.\n")
    judge = None if args.no_judge else get_llm_client()

    results = [
        compare(
            item,
            judge=judge,
            judge_sample=args.judge_sample,
            seed=args.seed,
        )
        for item in items
    ]
    totals = _summarize(results)

    out = args.out or _RESULTS / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"run_at": datetime.now().isoformat(), "backend": backend,
             "totals": totals, "results": results},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print(f"{'id':<4} {'ground +/-':<12} {'conf +/-':<10} {'cost +/-':<12} question")
    for r in results:
        a, b = r["with_verifier"], r["without_verifier"]

        def d(x, y, fmt):
            return fmt.format(x - y) if x is not None and y is not None else "-"

        print(f"{r['id']:<4} "
              f"{d(a['grounding_precision'], b['grounding_precision'], '{:+.2f}'):<12} "
              f"{d(a['overall_confidence'], b['overall_confidence'], '{:+.2f}'):<10} "
              f"{d(a['cost']['cost_usd'], b['cost']['cost_usd'], '{:+.4f}'):<12} "
              f"{r['question'][:44]}")
    print(f"\ngrounding precision delta: {totals['grounding_precision_delta']} "
          f"(with {totals['with_verifier']['avg_grounding_precision']} "
          f"vs without {totals['without_verifier']['avg_grounding_precision']})")
    print(f"total cost: ${totals['with_verifier']['total_cost_usd']:.4f} "
          f"(verifier on) vs ${totals['without_verifier']['total_cost_usd']:.4f} (off)")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
