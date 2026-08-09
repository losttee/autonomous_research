"""Memory store over the vector store + embedding client.

Two roles: recall a cached FinalReport for a near-identical question, and
surface past verified claims as MEMORY sources (internal RAG). On any error
the calls degrade to no-ops.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
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

# Skip writes when a near-identical entry exists.
_DEDUP_THRESHOLD = 0.98


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _age_days(ts: datetime) -> float:
    return (_now_utc() - ts).total_seconds() / 86400.0


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
        # `is None` on purpose: VectorStore defines __len__, so an empty
        # injected store is falsy and `store or ...` would drop it.
        path = f"{self._settings.chroma_persist_dir.rstrip('/')}/{_STORE_FILENAME}"
        self._store = store if store is not None else VectorStore(path)
        self._lock = threading.Lock()

    @property
    def vector_store(self) -> VectorStore:
        """The underlying store (for offline tooling)."""
        return self._store

    def _is_duplicate(self, vector: list[float], kind: str) -> bool:
        """True when a near-identical entry already exists (cosine >= dedup bar)."""
        hits = self._store.query(vector, top_k=1, kind=kind)
        return bool(hits) and hits[0][1] >= _DEDUP_THRESHOLD

    # --- Recall a whole report -------------------------------------------------

    def remember_report(
        self, report: FinalReport, tracker: Optional["CostTracker"] = None
    ) -> None:
        """Persist a report keyed by its question so a later run can reuse it.

        Near-identical questions already remembered are skipped.
        """
        try:
            vec = self._embedder.embed_one(report.question, tracker=tracker)
            if self._is_duplicate(vec, kind="report"):
                _log.info(
                    "report already remembered; skipping duplicate",
                    extra={"extra_fields": {"question": report.question[:80]}},
                )
                return
            self._store.add(
                text=report.question,
                vector=vec,
                metadata={
                    "kind": "report",
                    "report": report.model_dump(mode="json"),
                },
            )
        except Exception as exc:
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
        ttl = self._settings.memory_ttl_days
        if ttl > 0:
            age = _age_days(report.created_at)
            if age > ttl:
                log_step(
                    _log,
                    step_type="memory_recall",
                    step_id=report.report_id,
                    msg="matched report is stale; not reusing",
                    extra={"age_days": round(age, 1), "ttl_days": ttl},
                )
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
        """Persist each verified, supported claim as a retrievable memory entry.

        Claims already remembered (near-identical text) are skipped; every entry
        carries a timestamp so freshness can discount it later.
        """
        supported = [c for c in result.claims if c.supported and c.text]
        if not supported:
            return
        try:
            vecs = self._embedder.embed([c.text for c in supported], tracker=tracker)
            for claim, vec in zip(supported, vecs):
                if self._is_duplicate(vec, kind="claim"):
                    continue
                self._store.add(
                    text=claim.text,
                    vector=vec,
                    metadata={
                        "kind": "claim",
                        "confidence": claim.confidence,
                        "sub_task_id": result.sub_task_id,
                        "ts": _now_utc().isoformat(),
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

        Entries older than MEMORY_TTL_DAYS are dropped; the rest have their
        reliability decayed by up to half as they approach the TTL.
        Best-effort: any error -> []."""
        try:
            vec = self._embedder.embed_one(query, tracker=tracker)
            hits = self._store.query(vec, top_k=max_results, kind="claim")
        except Exception as exc:
            _log.warning(
                "search_memory failed", extra={"extra_fields": {"error": str(exc)}}
            )
            return []
        ttl = self._settings.memory_ttl_days
        sources: list[SourceRef] = []
        for entry, score in hits:
            if score < min_score:
                continue
            metadata = entry.get("metadata", {})
            base = float(metadata.get("confidence", 0.5))
            ts_raw = metadata.get("ts")
            age = 0.0  # legacy entries without a timestamp are treated as fresh
            if ts_raw:
                try:
                    age = _age_days(datetime.fromisoformat(ts_raw))
                except ValueError:
                    pass
            if ttl > 0 and age > ttl:
                continue  # stale memory is not served as evidence
            decay = max(0.5, 1.0 - 0.5 * age / ttl) if ttl > 0 else 1.0
            sources.append(
                SourceRef(
                    type=SourceType.MEMORY,
                    title="Recalled from memory",
                    snippet=entry.get("text", ""),
                    reliability=round(base * decay, 3),
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
    """No-op stand-in when memory can't initialize."""

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
