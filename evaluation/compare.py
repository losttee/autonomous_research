"""Side-by-side comparison of two evaluation runs.

    .venv\\Scripts\\python.exe -m evaluation.compare <old.json> <new.json>

Prints the totals and per-question deltas for the metrics that decide
whether a change improved or regressed the pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_KEYS = ("citation_integrity", "honesty_ok", "grounding_precision",
         "overall_confidence")


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value) -> str:
    if value is None:
        return "   -"
    if isinstance(value, bool):
        return "  ok" if value else "FAIL"
    return f"{value:.2f}"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip())
        return 2
    old, new = _load(sys.argv[1]), _load(sys.argv[2])

    print(f"{'metric':<26} {'old':>10} {'new':>10}")
    for key, ov in old["totals"].items():
        nv = new["totals"].get(key)
        if isinstance(ov, (int, float)) and not isinstance(ov, bool):
            print(f"{key:<26} {ov:>10} {nv:>10}")

    old_by_id = {r["id"]: r for r in old["results"]}
    print(f"\n{'id':<5} {'metric':<22} {'old':>6} {'new':>6}")
    for r in new["results"]:
        prev = old_by_id.get(r["id"])
        if prev is None:
            print(f"{r['id']:<5} (new question, not in old run)")
            continue
        for key in _KEYS:
            ov, nv = prev.get(key), r.get(key)
            if ov == nv:
                continue
            print(f"{r['id']:<5} {key:<22} {_fmt(ov):>6} {_fmt(nv):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
