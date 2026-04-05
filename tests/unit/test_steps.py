import pytest
from datetime import datetime

from src.pipeline.steps import (
    Query,
    RetrievalResult,
    SimilarityScore,
    TokenizationStep,
    EmbeddingStep,
    PipelineTrace,
    SimilarityStep,
    build_trace,
)


def make_embedding(dims: int = 384) -> list[float]:
    return [0.1] * dims


def make_scores(n: int = 3) -> list[SimilarityScore]:
    return [
        SimilarityScore(
            chunk_id=f"file.txt::{i}",
            chunk_text=f"Chunk text number {i}",
            chunk_embedding=make_embedding(),
            cosine_score=0.9 - i * 0.1,
            native_score=0.9 - i * 0.1,
            rank=i + 1,
        )
        for i in range(n)
    ]


def make_query() -> Query:
    return Query(
        text="How do I make a beurre blanc?",
        embedding=make_embedding(),
        tokens=["how", "do", "i", "make", "a", "beurre", "blanc", "?"],
        token_ids=[101, 2129, 2079, 1045, 2191, 1037, 6291, 21469, 1029, 102],
    )


def make_tokenization() -> TokenizationStep:
    return TokenizationStep(
        tokens=["how", "do", "i", "make"],
        token_ids=[101, 2129, 2079, 1045, 102],
        vocabulary_size=30522,
    )


def make_embedding_step() -> EmbeddingStep:
    return EmbeddingStep(
        embedding=make_embedding(),
        dimensions=384,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        inference_ms=42.0,
    )


def test_build_trace_returns_pipeline_trace():
    query = make_query()
    scores = make_scores()
    result = RetrievalResult(
        library_name="TestLib",
        query_text=query.text,
        scores=scores,
        latency_ms=15.0,
    )
    tokenization = make_tokenization()
    embedding = make_embedding_step()

    trace = build_trace(
        library_name="TestLib",
        query=query,
        result=result,
        tokenization=tokenization,
        embedding=embedding,
    )

    assert isinstance(trace, PipelineTrace)


def test_build_trace_has_correct_library_name():
    query = make_query()
    result = RetrievalResult(
        library_name="ChromaDB",
        query_text=query.text,
        scores=make_scores(),
        latency_ms=10.0,
    )
    trace = build_trace(
        library_name="ChromaDB",
        query=query,
        result=result,
        tokenization=make_tokenization(),
        embedding=make_embedding_step(),
    )
    assert trace.library_name == "ChromaDB"


def test_build_trace_query_text():
    query = make_query()
    result = RetrievalResult(
        library_name="FAISS",
        query_text=query.text,
        scores=make_scores(),
        latency_ms=5.0,
    )
    trace = build_trace(
        library_name="FAISS",
        query=query,
        result=result,
        tokenization=make_tokenization(),
        embedding=make_embedding_step(),
    )
    assert trace.query_text == query.text


def test_build_trace_all_fields_populated():
    query = make_query()
    scores = make_scores(5)
    result = RetrievalResult(
        library_name="Qdrant",
        query_text=query.text,
        scores=scores,
        latency_ms=8.0,
    )
    tokenization = make_tokenization()
    embedding = make_embedding_step()

    trace = build_trace(
        library_name="Qdrant",
        query=query,
        result=result,
        tokenization=tokenization,
        embedding=embedding,
    )

    assert trace.step_tokenization is not None
    assert trace.step_embedding is not None
    assert trace.step_similarity is not None
    assert trace.step_ranking is not None
    assert trace.step_tokenization.vocabulary_size == 30522
    assert trace.step_embedding.dimensions == 384
    assert trace.step_embedding.inference_ms == 42.0
    assert len(trace.step_similarity.candidate_scores) == 5
    assert trace.step_similarity.top_k == 5


def test_build_trace_cosine_scores_in_range():
    query = make_query()
    scores = make_scores(3)
    result = RetrievalResult(
        library_name="LanceDB",
        query_text=query.text,
        scores=scores,
        latency_ms=12.0,
    )
    trace = build_trace(
        library_name="LanceDB",
        query=query,
        result=result,
        tokenization=make_tokenization(),
        embedding=make_embedding_step(),
    )
    for s in trace.step_similarity.candidate_scores:
        assert -1.0 <= s.cosine_score <= 1.0
