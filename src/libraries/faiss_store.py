import faiss
import numpy as np
import time

from src.libraries.base import LibraryAdapter, IngestionResult, AdapterInitError
from src.pipeline.steps import EmbeddedChunk, RetrievalResult, SimilarityScore

DIMS = 384


class FAISSAdapter(LibraryAdapter):
    def __init__(self):
        self._index = None
        self._id_map: list[str] = []           # maps FAISS int id -> chunk id string
        self._text_map: dict[str, str] = {}    # chunk_id -> text
        self._emb_map: dict[str, list[float]] = {}  # chunk_id -> embedding

    @property
    def name(self) -> str:
        return "FAISS"

    def initialize(self) -> None:
        try:
            self._index = faiss.IndexFlatIP(DIMS)
        except Exception as e:
            raise AdapterInitError(f"FAISS initialization failed: {e}")

    def ingest(self, chunks: list[EmbeddedChunk]) -> IngestionResult:
        try:
            vecs = np.array([c.embedding for c in chunks], dtype=np.float32)
            faiss.normalize_L2(vecs)
            self._index.add(vecs)
            for c in chunks:
                self._id_map.append(c.id)
                self._text_map[c.id] = c.text
                self._emb_map[c.id] = c.embedding
            return IngestionResult(library_name=self.name, chunks_stored=len(chunks))
        except Exception as e:
            return IngestionResult(library_name=self.name, chunks_stored=0, error=str(e))

    def retrieve(self, query_embedding: list[float], top_k: int = 5, **_kwargs) -> RetrievalResult:
        start = time.time()
        try:
            q = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(q)
            k = min(top_k, self._index.ntotal)
            if k == 0:
                return RetrievalResult(library_name=self.name, query_text="", scores=[], latency_ms=0, error="No data ingested")
            distances, indices = self._index.search(q, k)
            scores = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(self._id_map):
                    continue
                cid = self._id_map[idx]
                scores.append(SimilarityScore(
                    chunk_id=cid,
                    chunk_text=self._text_map.get(cid, ""),
                    chunk_embedding=self._emb_map.get(cid, []),
                    cosine_score=float(dist),
                    native_score=float(dist),
                    rank=i + 1,
                ))
            latency = (time.time() - start) * 1000
            return RetrievalResult(library_name=self.name, query_text="", scores=scores, latency_ms=latency)
        except Exception as e:
            return RetrievalResult(library_name=self.name, query_text="", scores=[], latency_ms=0, error=str(e))

    def get_chunk_count(self) -> int:
        try:
            return self._index.ntotal if self._index else 0
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            self._index.reset()
            self._id_map.clear()
            self._text_map.clear()
            self._emb_map.clear()
        except Exception:
            pass

    def health_check(self) -> bool:
        return self._index is not None
