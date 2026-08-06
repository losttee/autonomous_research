"""Memory health probe — offline analysis of the vector store.

Measures what decides whether memory helps or hurts:

  - Store size per kind, with a migration nudge once it outgrows the
    pure-Python store (> 10k entries -> consider chromadb/redis).
  - Duplicate report clusters: pairs of stored questions with cosine
    similarity >= MEMORY_RECALL_THRESHOLD. Non-zero means recall should have
    short-circuited the second question but did not — a direct signal that
    the recall threshold or dedup needs tuning.

No LLM, no network — it only reads vectors already in the store.

    .venv\\Scripts\\python.exe -m evaluation.memory_probe
"""

from __future__ import annotations

import json
from typing import Any

from research_agent.core.config import get_settings
from research_agent.memory.store import get_memory_store
from research_agent.memory.vector_store import VectorStore, _cosine

_MIGRATION_THRESHOLD = 10_000


def analyze(store: VectorStore, recall_threshold: float) -> dict[str, Any]:
    """Roll the store up: sizes, per-kind counts, and duplicate report pairs."""
    entries = store.snapshot()
    reports = [e for e in entries if e.get("metadata", {}).get("kind") == "report"]
    claims = [e for e in entries if e.get("metadata", {}).get("kind") == "claim"]

    duplicate_pairs = 0
    for i, entry_a in enumerate(reports):
        vector_a = entry_a.get("vector", [])
        for entry_b in reports[i + 1:]:
            if _cosine(vector_a, entry_b.get("vector", [])) >= recall_threshold:
                duplicate_pairs += 1

    return {
        "entries": len(entries),
        "reports": len(reports),
        "claims": len(claims),
        "recall_threshold": recall_threshold,
        "duplicate_report_pairs": duplicate_pairs,
        "migration_suggested": len(entries) > _MIGRATION_THRESHOLD,
    }


def main() -> None:
    settings = get_settings()
    store_obj = get_memory_store()
    store = getattr(store_obj, "vector_store", None)
    if store is None:
        print(json.dumps({"entries": 0, "note": "memory is disabled or unavailable"}))
        return
    data = analyze(store, settings.memory_recall_threshold)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if data["duplicate_report_pairs"]:
        print(
            f"\nNOTE: {data['duplicate_report_pairs']} report pair(s) sit at/above the "
            "recall threshold — recall did not merge them. Consider raising "
            "MEMORY_RECALL_THRESHOLD or checking the dedup bar."
        )
    if data["migration_suggested"]:
        print(
            "\nNOTE: the pure-Python store is past ~10k entries — consider "
            "migrating to chromadb behind the same add()/query() surface."
        )


if __name__ == "__main__":
    main()
