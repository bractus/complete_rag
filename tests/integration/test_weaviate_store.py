import pytest

from src.libraries.base import AdapterInitError
from src.libraries.weaviate_store import WeaviateAdapter
from src.pipeline.steps import EmbeddedChunk


def make_chunks(n: int = 5) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            id=f"test.txt::{i}",
            source_filename="test.txt",
            chunk_index=i,
            text=f"This is chunk number {i} about French cuisine and cooking techniques.",
            embedding=[float(i % 10) / 10.0] * 383 + [float(i) / 100.0],
        )
        for i in range(n)
    ]


def make_query_embedding() -> list[float]:
    return [0.5] * 384


@pytest.fixture
def adapter():
    a = WeaviateAdapter()
    try:
        a.initialize()
    except AdapterInitError:
        pytest.skip("Weaviate embedded binary not available in this environment")
    yield a
    a.clear()


def test_initialize_succeeds(adapter):
    assert adapter.health_check() is True


def test_ingest_returns_correct_count(adapter):
    chunks = make_chunks(5)
    result = adapter.ingest(chunks)
    assert result.chunks_stored == 5
    assert result.error is None


def test_retrieve_top_3_after_ingest(adapter):
    chunks = make_chunks(10)
    adapter.ingest(chunks)
    query_emb = make_query_embedding()
    result = adapter.retrieve(query_emb, top_k=3)
    assert result.error is None
    assert len(result.scores) == 3


def test_retrieve_scores_in_valid_range(adapter):
    chunks = make_chunks(10)
    adapter.ingest(chunks)
    query_emb = make_query_embedding()
    result = adapter.retrieve(query_emb, top_k=3)
    for s in result.scores:
        assert -1.0 <= s.cosine_score <= 1.0, f"cosine_score {s.cosine_score} out of range"


def test_get_chunk_count(adapter):
    chunks = make_chunks(7)
    adapter.ingest(chunks)
    count = adapter.get_chunk_count()
    assert count == 7


def test_retrieve_returns_latency(adapter):
    adapter.ingest(make_chunks(5))
    result = adapter.retrieve(make_query_embedding(), top_k=3)
    assert result.latency_ms >= 0


def test_retrieve_scores_have_ranks(adapter):
    adapter.ingest(make_chunks(5))
    result = adapter.retrieve(make_query_embedding(), top_k=3)
    if result.scores:
        ranks = [s.rank for s in result.scores]
        assert ranks[0] == 1


def test_clear_resets_chunk_count(adapter):
    adapter.ingest(make_chunks(5))
    adapter.clear()
    assert adapter.get_chunk_count() == 0
