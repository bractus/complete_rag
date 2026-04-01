import lancedb
import numpy as np
import time
import shutil
import os

from src.libraries.base import LibraryAdapter, IngestionResult, AdapterInitError
from src.pipeline.steps import EmbeddedChunk, RetrievalResult, SimilarityScore

DB_PATH = ".lancedb_tmp"
TABLE_NAME = "rag_comparison"


class LanceDBAdapter(LibraryAdapter):
    def __init__(self):
        self._db = None
        self._table = None
        self._chunks: dict[str, EmbeddedChunk] = {}

    @property
    def name(self) -> str:
        return "LanceDB"

    def initialize(self) -> None:
        try:
            if os.path.exists(DB_PATH):
                shutil.rmtree(DB_PATH, ignore_errors=True)
            self._db = lancedb.connect(DB_PATH)
            self._table = None
        except Exception as e:
            raise AdapterInitError(f"LanceDB initialization failed: {e}")

    def ingest(self, chunks: list[EmbeddedChunk]) -> IngestionResult:
        try:
            data = [
                {"id": c.id, "text": c.text, "vector": c.embedding}
                for c in chunks
            ]
            if self._table is None:
                self._table = self._db.create_table(TABLE_NAME, data=data, mode="overwrite")
            else:
                self._table.add(data)
            for c in chunks:
                self._chunks[c.id] = c
            return IngestionResult(library_name=self.name, chunks_stored=len(chunks))
        except Exception as e:
            return IngestionResult(library_name=self.name, chunks_stored=0, error=str(e))

    def retrieve(self, query_embedding: list[float], top_k: int = 5, **kwargs) -> RetrievalResult:
        start = time.time()
        try:
            if self._table is None:
                return RetrievalResult(library_name=self.name, query_text="", scores=[], latency_ms=0, error="No data ingested")
            metric = kwargs.get("metric", "cosine")
            results = self._table.search(query_embedding).metric(metric).limit(top_k).to_list()
            scores = []
            for i, row in enumerate(results):
                cid = row.get("id", "")
                text = row.get("text", "")
                vec = row.get("vector", [])
                # LanceDB returns L2 distance by default; compute cosine from stored embeddings
                q = np.array(query_embedding)
                v = np.array(vec) if vec else np.zeros(len(query_embedding))
                denom = (np.linalg.norm(q) * np.linalg.norm(v))
                cosine = float(np.dot(q, v) / denom) if denom > 0 else 0.0
                native = row.get("_distance", 0.0)
                scores.append(SimilarityScore(
                    chunk_id=cid,
                    chunk_text=text,
                    chunk_embedding=list(vec) if vec else [],
                    cosine_score=cosine,
                    native_score=native,
                    rank=i + 1,
                ))
            # Re-sort by cosine_score descending
            scores.sort(key=lambda s: s.cosine_score, reverse=True)
            for i, s in enumerate(scores):
                s.rank = i + 1
            latency = (time.time() - start) * 1000
            return RetrievalResult(library_name=self.name, query_text="", scores=scores, latency_ms=latency)
        except Exception as e:
            return RetrievalResult(library_name=self.name, query_text="", scores=[], latency_ms=0, error=str(e))

    def get_chunk_count(self) -> int:
        try:
            return len(self._table) if self._table is not None else 0
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            if self._db:
                self._db.drop_table(TABLE_NAME, ignore_missing=True)
                self._table = None
                self._chunks.clear()
        except Exception:
            pass

    def health_check(self) -> bool:
        return self._db is not None
