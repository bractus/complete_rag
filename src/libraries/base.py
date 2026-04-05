from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.pipeline.steps import EmbeddedChunk, RetrievalResult


class AdapterInitError(Exception):
    pass


@dataclass
class IngestionResult:
    library_name: str
    chunks_stored: int
    error: Optional[str] = None


class LibraryAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def initialize(self) -> None: ...  # ONLY raises AdapterInitError

    @abstractmethod
    def ingest(self, chunks: list[EmbeddedChunk]) -> IngestionResult: ...  # never raises

    @abstractmethod
    def retrieve(self, query_embedding: list[float], top_k: int = 5, **kwargs) -> RetrievalResult: ...  # never raises

    @abstractmethod
    def get_chunk_count(self) -> int: ...  # returns 0 on error

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def health_check(self) -> bool: ...
