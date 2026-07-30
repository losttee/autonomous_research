"""Aggregate JSON-lines pipeline logs into dashboard-ready numbers.

Pure functions over a file — no server state, trivially testable. Corrupt or
partial lines are skipped silently: the dashboard must stay up no matter what
ends up in the log.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# research_done lines mark finished runs and carry the run-level extras.
_RUN_STEP = "research_done"
_RECENT_RUNS = 20


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield each parseable JSON object in the log file; skip the rest."""
    log_path = Path(path)
    if not log_path.exists():
        return
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def aggregate(path: str | Path) -> dict[str, Any]:
    """Roll the log up: totals, per-step stats, and the most recent runs."""
    by_step: dict[str, dict[str, Any]] = {}
    runs: list[dict[str, Any]] = []
    totals = {"steps": 0, "tokens": 0, "cost_usd": 0.0, "budget_exceeded": 0}

    for record in iter_records(path):
        step = str(record.get("step_type") or "_other")
        latency = int(record.get("latency_ms") or 0)
        tokens = int(record.get("tokens") or 0)
        cost = float(record.get("cost_usd") or 0.0)

        bucket = by_step.setdefault(
            step, {"count": 0, "latency_ms": 0, "tokens": 0, "cost_usd": 0.0}
        )
        bucket["count"] += 1
        bucket["latency_ms"] += latency
        bucket["tokens"] += tokens
        bucket["cost_usd"] += cost

        totals["steps"] += 1
        totals["tokens"] += tokens
        totals["cost_usd"] += cost
        if step == "budget_exceeded":
            totals["budget_exceeded"] += 1
        if step == _RUN_STEP:
            runs.append(
                {
                    "report_id": record.get("step_id", ""),
                    "ts": record.get("ts", ""),
                    "latency_ms": latency,
                    "cost_usd": cost,
                    "sections": record.get("sections"),
                    "sources": record.get("sources"),
                    "revision": record.get("revision", 0),
                    "recalled": bool(record.get("recalled")),
                }
            )

    runs.sort(key=lambda r: r["ts"], reverse=True)
    step_rows = [
        {
            "step": name,
            "count": bucket["count"],
            "avg_latency_ms": round(bucket["latency_ms"] / bucket["count"], 1),
            "tokens": bucket["tokens"],
            "cost_usd": round(bucket["cost_usd"], 6),
        }
        for name, bucket in by_step.items()
    ]
    step_rows.sort(key=lambda row: row["count"], reverse=True)

    run_count = len(runs)
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": totals,
        "run_count": run_count,
        "avg_run_latency_ms": (
            round(sum(r["latency_ms"] for r in runs) / run_count, 1) if run_count else 0.0
        ),
        "avg_run_cost_usd": (
            round(sum(r["cost_usd"] for r in runs) / run_count, 6) if run_count else 0.0
        ),
        "by_step": step_rows,
        "runs": runs[:_RECENT_RUNS],
    }
