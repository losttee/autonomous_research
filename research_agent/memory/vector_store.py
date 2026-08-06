"""Pure-Python vector store — the persistence layer under memory/.

Deliberately dependency-free (no chromadb/numpy): the corpus here is small
(reports + claims from past runs on one machine), so a JSON file plus cosine
similarity in plain Python is enough and keeps a fresh checkout runnable with no
native builds. Swap this for chromadb later behind the same add()/query() surface
if the corpus outgrows an in-memory scan.

Thread-safe: the pipeline runs sub-tasks on a thread pool (see cost_tracker.py),
so multiple workers may add()/query() at once.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 on degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class VectorStore:
    """In-memory vectors backed by a single JSON file, queried by cosine similarity.

    Each entry: {id, text, vector, metadata, ts}. Loaded lazily on construction;
    every add() flushes to disk so state survives across process restarts.
    """

    def __init__(self, persist_path: str | Path) -> None:
        self._path = Path(persist_path)
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self._entries = [e for e in data if isinstance(e, dict)]
        except (json.JSONDecodeError, OSError):
            # Corrupt/unreadable store must not crash a run — start empty.
            self._entries = []

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._entries, fh, ensure_ascii=False)
        tmp.replace(self._path)  # atomic-ish swap, avoids half-written files

    def add(
        self,
        text: str,
        vector: list[float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add one entry and persist. Returns the entry id."""
        entry_id = f"mem_{uuid4().hex[:12]}"
        with self._lock:
            self._entries.append(
                {
                    "id": entry_id,
                    "text": text,
                    "vector": list(vector),
                    "metadata": metadata or {},
                }
            )
            self._flush()
        return entry_id

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        kind: Optional[str] = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Return up to top_k (entry, score) pairs, highest cosine first.

        kind filters on metadata['kind'] (e.g. 'report' vs 'claim') so recall and
        RAG search can share one store without cross-contaminating results.
        """
        with self._lock:
            snapshot = list(self._entries)
        scored: list[tuple[dict[str, Any], float]] = []
        for entry in snapshot:
            if kind is not None and entry.get("metadata", {}).get("kind") != kind:
                continue
            scored.append((entry, _cosine(vector, entry.get("vector", []))))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def snapshot(self) -> list[dict[str, Any]]:
        """Shallow copies of all entries — a read-only view for offline tooling
        (e.g. the evaluation memory probe)."""
        with self._lock:
            return [dict(entry) for entry in self._entries]
