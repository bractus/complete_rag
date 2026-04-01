from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Document:
    filename: str
    content: str
    size_bytes: int
    upload_ts: datetime = field(default_factory=datetime.now)


@dataclass
class Chunk:
    id: str           # format: "{filename}::{index}"
    source_filename: str
    chunk_index: int
    text: str


@dataclass
class EmbeddedChunk:
    id: str
    source_filename: str
    chunk_index: int
    text: str
    embedding: list[float]


@dataclass
class Query:
    text: str
    embedding: list[float]
    tokens: list[str]
    token_ids: list[int]
    submitted_ts: datetime = field(default_factory=datetime.now)


@dataclass
class SimilarityScore:
    chunk_id: str
    chunk_text: str
    chunk_embedding: list[float]
    cosine_score: float
    native_score: float
    rank: int


@dataclass
class RetrievalResult:
    library_name: str
    query_text: str
    scores: list[SimilarityScore]
    latency_ms: float
    error: Optional[str] = None


@dataclass
class TokenizationStep:
    tokens: list[str]
    token_ids: list[int]
    vocabulary_size: int


@dataclass
class EmbeddingStep:
    embedding: list[float]
    dimensions: int
    model_name: str
    inference_ms: float


@dataclass
class SimilarityStep:
    candidate_scores: list[SimilarityScore]
    top_k: int


@dataclass
class PipelineTrace:
    library_name: str
    query_text: str
    step_tokenization: TokenizationStep
    step_embedding: EmbeddingStep
    step_similarity: SimilarityStep
    step_ranking: RetrievalResult


@dataclass
class LibraryStatus:
    name: str
    enabled: bool = True
    connected: bool = False
    chunk_count: int = 0
    last_ingest_ts: Optional[datetime] = None
    error: Optional[str] = None


def build_trace(
    library_name: str,
    query: Query,
    result: RetrievalResult,
    tokenization: TokenizationStep,
    embedding: EmbeddingStep,
) -> PipelineTrace:
    # Build SimilarityStep from result scores (already computed cosine)
    similarity_step = SimilarityStep(
        candidate_scores=result.scores,
        top_k=len(result.scores),
    )
    return PipelineTrace(
        library_name=library_name,
        query_text=query.text,
        step_tokenization=tokenization,
        step_embedding=embedding,
        step_similarity=similarity_step,
        step_ranking=result,
    )
