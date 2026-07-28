"""Memory & Vector Store tests — recall, internal RAG, and offline fallbacks.

Run: .venv/Scripts/python.exe -m pytest tests/test_memory.py -v
"""

from __future__ import annotations

from research_agent.core.config import get_settings
from research_agent.core.contracts import (
    Claim,
    FinalReport,
    SourceRef,
    SourceType,
    SubTaskResult,
    SubTaskStatus,
)
from research_agent.core.embeddings import EmbeddingClient, _hash_embedding
from research_agent.guardrail.cost_tracker import CostTracker
from research_agent.memory.store import MemoryStore
from research_agent.memory.vector_store import VectorStore, _cosine


class FakeEmbedder:
    """Deterministic embedder using the offline hashing embedding — no network."""

    def embed(self, texts, tracker=None):
        return [_hash_embedding(t) for t in texts]

    def embed_one(self, text, tracker=None):
        return _hash_embedding(text)


def _mem(tmp_path) -> MemoryStore:
    store = VectorStore(tmp_path / "mem.json")
    return MemoryStore(embedder=FakeEmbedder(), store=store)


def test_cosine_ranks_similar_text_higher() -> None:
    q = _hash_embedding("premium of insurance plan A")
    near = _hash_embedding("insurance plan A premium")
    far = _hash_embedding("weather forecast tomorrow")
    assert _cosine(q, near) > _cosine(q, far)


def test_vector_store_persists_and_reloads(tmp_path) -> None:
    path = tmp_path / "mem.json"
    vs = VectorStore(path)
    vs.add("hello world", _hash_embedding("hello world"), {"kind": "claim"})
    assert len(vs) == 1
    # A fresh store over the same file must see the persisted entry.
    reopened = VectorStore(path)
    assert len(reopened) == 1
    hits = reopened.query(_hash_embedding("hello world"), top_k=1, kind="claim")
    assert hits and hits[0][1] > 0.9


def test_recall_hit_above_threshold(tmp_path) -> None:
    mem = _mem(tmp_path)
    report = FinalReport(
        question="Compare plan A and plan B by premium",
        plan_id="plan_x",
        recommendation="Plan A is cheaper.",
        overall_confidence=0.7,
    )
    mem.remember_report(report)
    hit = mem.recall_report("Compare plan A and plan B by premium")
    assert hit is not None, "identical question must recall the report"
    recalled, score = hit
    assert recalled.recommendation == "Plan A is cheaper."
    assert score >= get_settings().memory_recall_threshold


def test_recall_miss_for_unrelated_question(tmp_path) -> None:
    mem = _mem(tmp_path)
    mem.remember_report(
        FinalReport(question="premium of plan A", plan_id="p", recommendation="x")
    )
    assert mem.recall_report("how tall is Mount Everest") is None


def test_search_memory_returns_memory_sources(tmp_path) -> None:
    mem = _mem(tmp_path)
    result = SubTaskResult(
        sub_task_id="t1",
        status=SubTaskStatus.DONE,
        claims=[
            Claim(text="Plan A premium is $10/mo", supporting_source_ids=["src_1"],
                  supported=True, confidence=0.8)
        ],
        sources=[SourceRef(type=SourceType.WEB, source_id="src_1", snippet="...")],
    )
    mem.remember_claims(result)
    sources = mem.search_memory("Plan A premium is $10/mo", min_score=0.5)
    assert sources, "a near-identical query should recall the stored claim"
    assert all(s.type == SourceType.MEMORY for s in sources)


def test_unsupported_claims_are_not_remembered(tmp_path) -> None:
    mem = _mem(tmp_path)
    result = SubTaskResult(
        sub_task_id="t1",
        status=SubTaskStatus.DONE,
        claims=[Claim(text="unverified", supporting_source_ids=["src_1"], supported=False)],
        sources=[SourceRef(type=SourceType.WEB, source_id="src_1")],
    )
    mem.remember_claims(result)
    assert mem.search_memory("unverified", min_score=0.5) == []


def test_embedding_falls_back_offline_on_api_error() -> None:
    client = EmbeddingClient(settings=get_settings().model_copy())
    client._settings.llm_api_key = "x"  # force the real path to be attempted

    class Boom:
        class embeddings:
            @staticmethod
            def create(*a, **k):
                raise RuntimeError("no /embeddings on this gateway")

    client._client = Boom()
    vecs = client.embed(["hello", "world"], tracker=CostTracker())
    assert len(vecs) == 2 and len(vecs[0]) == 256
    assert client.degraded, "client should mark itself degraded after a failure"


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    test_cosine_ranks_similar_text_higher()
    test_vector_store_persists_and_reloads(tmp)
    test_recall_hit_above_threshold(tmp)
    test_recall_miss_for_unrelated_question(tmp)
    test_search_memory_returns_memory_sources(tmp)
    test_unsupported_claims_are_not_remembered(tmp)
    test_embedding_falls_back_offline_on_api_error()
    print("MEMORY TESTS PASSED")
    sys.exit(0)
