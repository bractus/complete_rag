import chromadb
import numpy as np
import time

from src.libraries.base import LibraryAdapter, IngestionResult, AdapterInitError
from src.pipeline.steps import EmbeddedChunk, RetrievalResult, SimilarityScore


class ChromaDBAdapter(LibraryAdapter):
    def __init__(self):
        self._client = None
        self._collection = None
        self._chunks: dict[str, EmbeddedChunk] = {}

    @property
    def name(self) -> str:
        return "ChromaDB"

    def initialize(self) -> None:
        try:
            self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(
                name="rag_comparison",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise AdapterInitError(f"ChromaDB initialization failed: {e}")

    def ingest(self, chunks: list[EmbeddedChunk]) -> IngestionResult:
        try:
            ids = [c.id for c in chunks]
            documents = [c.text for c in chunks]
            embeddings = [c.embedding for c in chunks]
            self._collection.upsert(ids=ids, documents=documents, embeddings=embeddings)
            for c in chunks:
                self._chunks[c.id] = c
            return IngestionResult(library_name=self.name, chunks_stored=len(chunks))
        except Exception as e:
            return IngestionResult(library_name=self.name, chunks_stored=0, error=str(e))

    def retrieve(self, query_embedding: list[float], top_k: int = 5, **_kwargs) -> RetrievalResult:
        start = time.time()
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.get_chunk_count()),
                include=["documents", "distances", "embeddings"],
            )
            scores = []
            ids = results["ids"][0]
            docs = results["documents"][0]
            distances = results["distances"][0]
            embs = results.get("embeddings") or [[None] * len(ids)]
            embs = embs[0] if embs else [None] * len(ids)
            for i, (cid, doc, dist) in enumerate(zip(ids, docs, distances)):
                # ChromaDB cosine distance = 1 - cosine_similarity
                cosine = 1.0 - dist
                chunk_emb = list(embs[i]) if embs[i] is not None else []
                if not chunk_emb and cid in self._chunks:
                    chunk_emb = self._chunks[cid].embedding
                scores.append(SimilarityScore(
                    chunk_id=cid,
                    chunk_text=doc,
                    chunk_embedding=chunk_emb,
                    cosine_score=cosine,
                    native_score=dist,
                    rank=i + 1,
                ))
            latency = (time.time() - start) * 1000
            return RetrievalResult(library_name=self.name, query_text="", scores=scores, latency_ms=latency)
        except Exception as e:
            return RetrievalResult(library_name=self.name, query_text="", scores=[], latency_ms=0, error=str(e))

    def get_chunk_count(self) -> int:
        try:
            return self._collection.count() if self._collection else 0
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            if self._client and self._collection:
                self._client.delete_collection("rag_comparison")
                self._collection = self._client.get_or_create_collection(
                    name="rag_comparison",
                    metadata={"hnsw:space": "cosine"},
                )
                self._chunks.clear()
        except Exception:
            pass

    def health_check(self) -> bool:
        try:
            return self._collection is not None
        except Exception:
            return False
