"""Memory maturity tests: dedup, freshness/TTL, recall decay, and the probe.

Deterministic: the offline hashing embedder, temp stores, no network.
Run: .venv\\Scripts\\python.exe -m pytest tests/test_memory_maturity.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_agent.core.contracts import (
    Claim,
    FinalReport,
    SourceRef,
    SourceType,
    SubTaskResult,
    SubTaskStatus,
)
from research_agent.core.embeddings import _hash_embedding
from research_agent.memory.store import MemoryStore
from research_agent.memory.vector_store import VectorStore

from evaluation.memory_probe import analyze


class FakeEmbedder:
    """Deterministic embedder using the offline hashing embedding (no network)."""

    def embed(self, texts, tracker=None):
        return [_hash_embedding(t) for t in texts]

    def embed_one(self, text, tracker=None):
        return _hash_embedding(text)


def _mem(tmp_path) -> MemoryStore:
    return MemoryStore(embedder=FakeEmbedder(), store=VectorStore(tmp_path / "mem.json"))


def _claim_entry(tmp_path, text: str, ts: str | None, confidence: float = 0.8) -> MemoryStore:
    """A store pre-seeded with one claim entry carrying an explicit timestamp."""
    vs = VectorStore(tmp_path / "mem.json")
    metadata: dict = {"kind": "claim", "confidence": confidence}
    if ts is not None:
        metadata["ts"] = ts
    vs.add(text=text, vector=_hash_embedding(text), metadata=metadata)
    return MemoryStore(embedder=FakeEmbedder(), store=vs)


def _ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


# --- dedup ---------------------------------------------------------------------


def test_remember_claims_skips_duplicates(tmp_path) -> None:
    mem = _mem(tmp_path)
    result = SubTaskResult(
        sub_task_id="t1",
        status=SubTaskStatus.DONE,
        claims=[Claim(text="Plan A premium is $10/mo", supporting_source_ids=["src_1"],
                      supported=True, confidence=0.8)],
        sources=[SourceRef(type=SourceType.WEB, source_id="src_1", snippet="...")],
    )

    mem.remember_claims(result)
    mem.remember_claims(result)  # re-running the same topic must not duplicate

    assert len(mem.vector_store) == 1


def test_remember_report_skips_near_identical_question(tmp_path) -> None:
    mem = _mem(tmp_path)
    report = FinalReport(
        question="Compare plan A and plan B by premium",
        plan_id="plan_x",
        recommendation="Plan A is cheaper.",
    )

    mem.remember_report(report)
    mem.remember_report(report)

    assert len(mem.vector_store) == 1


def test_distinct_memories_are_both_kept(tmp_path) -> None:
    mem = _mem(tmp_path)
    for text in ("premium of insurance plan A", "weather forecast for Hanoi tomorrow"):
        result = SubTaskResult(
            sub_task_id="t",
            status=SubTaskStatus.DONE,
            claims=[Claim(text=text, supporting_source_ids=["src_1"],
                          supported=True, confidence=0.8)],
            sources=[SourceRef(type=SourceType.WEB, source_id="src_1", snippet="...")],
        )
        mem.remember_claims(result)

    assert len(mem.vector_store) == 2


# --- freshness / TTL --------------------------------------------------------------


def test_recall_refuses_stale_report(tmp_path) -> None:
    mem = _mem(tmp_path)
    stale = FinalReport(
        question="Compare plan A and plan B by premium",
        plan_id="plan_x",
        recommendation="Plan A is cheaper.",
        created_at=datetime.now(timezone.utc) - timedelta(days=40),
    )
    mem.remember_report(stale)

    assert mem.recall_report("Compare plan A and plan B by premium") is None, \
        "a 40-day-old report must not be served with the default 30-day TTL"


def test_recall_serves_fresh_report(tmp_path) -> None:
    mem = _mem(tmp_path)
    fresh = FinalReport(
        question="Compare plan A and plan B by premium",
        plan_id="plan_x",
        recommendation="Plan A is cheaper.",
    )
    mem.remember_report(fresh)

    hit = mem.recall_report("Compare plan A and plan B by premium")
    assert hit is not None and hit[0].recommendation == "Plan A is cheaper."


def test_search_memory_skips_expired_claims(tmp_path) -> None:
    mem = _claim_entry(tmp_path, "Plan A premium is $10/mo", ts=_ago(40))
    assert mem.search_memory("Plan A premium is $10/mo", min_score=0.5) == []


def test_search_memory_decays_stale_claim_reliability(tmp_path) -> None:
    mem = _claim_entry(tmp_path, "Plan A premium is $10/mo", ts=_ago(15), confidence=0.8)
    sources = mem.search_memory("Plan A premium is $10/mo", min_score=0.5)
    assert sources
    # Half the TTL -> reliability decays by a quarter: 0.8 * 0.75 = 0.6
    assert abs(sources[0].reliability - 0.6) < 0.02


def test_search_memory_keeps_full_trust_for_fresh_and_legacy(tmp_path) -> None:
    fresh = _claim_entry(tmp_path, "Plan A premium is $10/mo", ts=datetime.now(timezone.utc).isoformat())
    assert fresh.search_memory("Plan A premium is $10/mo", min_score=0.5)[0].reliability == 0.8

    legacy = _claim_entry(tmp_path, "Plan A premium is $10/mo", ts=None)
    assert legacy.search_memory("Plan A premium is $10/mo", min_score=0.5)[0].reliability == 0.8, \
        "entries saved before timestamps existed stay fully trusted"


# --- probe --------------------------------------------------------------------------


def test_probe_counts_duplicate_report_pairs(tmp_path) -> None:
    vs = VectorStore(tmp_path / "mem.json")
    question = "compare plan A and plan B"
    vs.add(text=question, vector=_hash_embedding(question), metadata={"kind": "report"})
    vs.add(text=question, vector=_hash_embedding(question), metadata={"kind": "report"})
    vs.add(text="unrelated question entirely", vector=_hash_embedding("unrelated question entirely"),
           metadata={"kind": "report"})
    vs.add(text="a claim", vector=_hash_embedding("a claim"), metadata={"kind": "claim"})

    data = analyze(vs, recall_threshold=0.92)
    assert data["entries"] == 4
    assert data["reports"] == 3 and data["claims"] == 1
    assert data["duplicate_report_pairs"] == 1
    assert data["migration_suggested"] is False


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    test_remember_claims_skips_duplicates(tmp)
    test_remember_report_skips_near_identical_question(tmp)
    test_distinct_memories_are_both_kept(tmp)
    test_recall_refuses_stale_report(tmp)
    test_recall_serves_fresh_report(tmp)
    test_search_memory_skips_expired_claims(tmp)
    test_search_memory_decays_stale_claim_reliability(tmp)
    test_search_memory_keeps_full_trust_for_fresh_and_legacy(tmp)
    test_probe_counts_duplicate_report_pairs(tmp)
    print("MEMORY MATURITY TESTS PASSED")
