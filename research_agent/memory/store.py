"""Memory store — the business layer over the vector store + embedding client.

Two key roles:
  1. Recall: cache a whole FinalReport keyed by its question; on a near-identical
     question later, return it instead of paying to research again.
  2. Internal RAG: expose past verified claims as MEMORY SourceRefs so the executor
     can cite prior findings alongside fresh web results.

Everything degrades: if embeddings/store fail, calls become no-ops (recall misses,
RAG returns nothing) and a research run proceeds exactly as if memory were off.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

from research_agent.core.config import Settings, get_settings
from research_agent.core.contracts import (
    FinalReport,
    SourceRef,
    SourceType,
    SubTaskResult,
)
from research_agent.core.embeddings import EmbeddingClient, get_embedding_client
from research_agent.core.logging import get_logger, log_step
from research_agent.memory.vector_store import VectorStore

if TYPE_CHECKING:
    from research_agent.guardrail.cost_tracker import CostTracker

_log = get_logger("memory")

_STORE_FILENAME = "memory.json"


class MemoryStore:
    """Recall past reports and surface past claims as RAG sources."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedder: Optional[EmbeddingClient] = None,
        store: Optional[VectorStore] = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._embedder = embedder if embedder is not None else get_embedding_client()
        # NOTE: `is None`, not `or` — VectorStore defines __len__, so an empty
        # injected store is falsy and `store or ...` would silently discard it.
        # TODO: pure-Python vector search is fast enough for <10k claims; swap with Redis/Chroma if scaled.
        path = f"{self._settings.chroma_persist_dir.rstrip('/')}/{_STORE_FILENAME}"
        self._store = store if store is not None else VectorStore(path)
        self._lock = threading.Lock()

    # --- Recall a whole report -------------------------------------------------

    def remember_report(
        self, report: FinalReport, tracker: Optional["CostTracker"] = None
    ) -> None:
        """Persist a report keyed by its question so a later run can reuse it."""
        try:
            vec = self._embedder.embed_one(report.question, tracker=tracker)
            self._store.add(
                text=report.question,
                vector=vec,
                metadata={
                    "kind": "report",
                    "report": report.model_dump(mode="json"),
                },
            )
        except Exception as exc:  # memory is best-effort; never break the run
            _log.warning(
                "remember_report failed",
                extra={"extra_fields": {"error": str(exc)}},
            )

    def recall_report(
        self, question: str, tracker: Optional["CostTracker"] = None
    ) -> Optional[tuple[FinalReport, float]]:
        """Return a cached report + score when a near-identical question was seen
        before (score >= threshold), else None. Best-effort: any error -> miss."""
        try:
            vec = self._embedder.embed_one(question, tracker=tracker)
            hits = self._store.query(vec, top_k=1, kind="report")
        except Exception as exc:
            _log.warning(
                "recall_report failed", extra={"extra_fields": {"error": str(exc)}}
            )
            return None
        if not hits:
            return None
        entry, score = hits[0]
        if score < self._settings.memory_recall_threshold:
            return None
        raw = entry.get("metadata", {}).get("report")
        if not isinstance(raw, dict):
            return None
        try:
            report = FinalReport.model_validate(raw)
        except Exception:
            return None
        log_step(
            _log,
            step_type="memory_recall",
            step_id=report.report_id,
            msg="recalled report from memory",
            extra={"score": round(score, 4), "question": question[:80]},
        )
        return report, score

    # --- Internal RAG: past claims as sources ---------------------------------

    def remember_claims(
        self, result: SubTaskResult, tracker: Optional["CostTracker"] = None
    ) -> None:
        """Persist each verified, supported claim as a retrievable memory entry."""
        supported = [c for c in result.claims if c.supported and c.text]
        if not supported:
            return
        try:
            vecs = self._embedder.embed([c.text for c in supported], tracker=tracker)
            for claim, vec in zip(supported, vecs):
                self._store.add(
                    text=claim.text,
                    vector=vec,
                    metadata={
                        "kind": "claim",
                        "confidence": claim.confidence,
                        "sub_task_id": result.sub_task_id,
                    },
                )
        except Exception as exc:
            _log.warning(
                "remember_claims failed",
                extra={"extra_fields": {"error": str(exc)}},
            )

    def search_memory(
        self,
        query: str,
        tracker: Optional["CostTracker"] = None,
        max_results: int = 3,
        min_score: float = 0.6,
    ) -> list[SourceRef]:
        """Return past claims relevant to the query as MEMORY SourceRefs.

        These are grounded (they were supported when stored), so the executor can
        cite them alongside fresh web results. Best-effort: any error -> []."""
        try:
            vec = self._embedder.embed_one(query, tracker=tracker)
            hits = self._store.query(vec, top_k=max_results, kind="claim")
        except Exception as exc:
            _log.warning(
                "search_memory failed", extra={"extra_fields": {"error": str(exc)}}
            )
            return []
        sources: list[SourceRef] = []
        for entry, score in hits:
            if score < min_score:
                continue
            sources.append(
                SourceRef(
                    type=SourceType.MEMORY,
                    title="Recalled from memory",
                    snippet=entry.get("text", ""),
                    reliability=float(entry.get("metadata", {}).get("confidence", 0.5)),
                )
            )
        if sources:
            log_step(
                _log,
                step_type="memory_search",
                step_id="memory",
                msg="recalled claims as RAG sources",
                extra={"query": query[:80], "results": len(sources)},
            )
        return sources


class _NullMemoryStore:
    """No-op stand-in when memory can't initialize — every call degrades safely."""

    def remember_report(self, report, tracker=None) -> None:  # noqa: D401
        return None

    def recall_report(self, question, tracker=None):
        return None

    def remember_claims(self, result, tracker=None) -> None:
        return None

    def search_memory(self, query, tracker=None, max_results=3, min_score=0.6):
        return []


_store: Optional[MemoryStore | _NullMemoryStore] = None


def get_memory_store() -> MemoryStore | _NullMemoryStore:
    """Singleton memory store; a no-op store if initialization fails."""
    global _store
    if _store is None:
        try:
            _store = MemoryStore()
        except Exception as exc:
            _log.warning(
                "memory store unavailable; running without memory",
                extra={"extra_fields": {"error": str(exc)}},
            )
            _store = _NullMemoryStore()
    return _store
