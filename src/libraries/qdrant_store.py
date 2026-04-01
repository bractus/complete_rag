import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.libraries.base import LibraryAdapter, IngestionResult, AdapterInitError
from src.pipeline.steps import EmbeddedChunk, RetrievalResult, SimilarityScore

COLLECTION = "rag_comparison"
DIMS = 384


class QdrantAdapter(LibraryAdapter):
    def __init__(self):
        self._client = None
        self._chunks: dict[str, EmbeddedChunk] = {}
        self._id_to_point: dict[str, int] = {}  # chunk_id -> numeric point id
        self._point_to_id: dict[int, str] = {}
        self._counter = 0

    @property
    def name(self) -> str:
        return "Qdrant"

    def initialize(self) -> None:
        try:
            self._client = QdrantClient(":memory:")
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE),
            )
        except Exception as e:
            raise AdapterInitError(f"Qdrant initialization failed: {e}")

    def ingest(self, chunks: list[EmbeddedChunk]) -> IngestionResult:
        try:
            points = []
            for c in chunks:
                if c.id not in self._id_to_point:
                    pid = self._counter
                    self._counter += 1
                    self._id_to_point[c.id] = pid
                    self._point_to_id[pid] = c.id
                else:
                    pid = self._id_to_point[c.id]
                points.append(PointStruct(
                    id=pid,
                    vector=c.embedding,
                    payload={"chunk_id": c.id, "text": c.text},
                ))
                self._chunks[c.id] = c
            self._client.upsert(collection_name=COLLECTION, points=points)
            return IngestionResult(library_name=self.name, chunks_stored=len(chunks))
        except Exception as e:
            return IngestionResult(library_name=self.name, chunks_stored=0, error=str(e))

    def retrieve(self, query_embedding: list[float], top_k: int = 5, **kwargs) -> RetrievalResult:
        start = time.time()
        try:
            score_threshold = kwargs.get("score_threshold", 0.0) or None
            response = self._client.query_points(
                collection_name=COLLECTION,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=True,
            )
            scores = []
            for i, hit in enumerate(response.points):
                cid = hit.payload.get("chunk_id", str(hit.id))
                text = hit.payload.get("text", "")
                emb = list(hit.vector) if hit.vector else []
                scores.append(SimilarityScore(
                    chunk_id=cid,
                    chunk_text=text,
                    chunk_embedding=emb,
                    cosine_score=float(hit.score),
                    native_score=float(hit.score),
                    rank=i + 1,
                ))
            latency = (time.time() - start) * 1000
            return RetrievalResult(library_name=self.name, query_text="", scores=scores, latency_ms=latency)
        except Exception as e:
            return RetrievalResult(library_name=self.name, query_text="", scores=[], latency_ms=0, error=str(e))

    def get_chunk_count(self) -> int:
        try:
            info = self._client.get_collection(COLLECTION)
            return info.points_count or 0
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            self._client.delete_collection(COLLECTION)
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE),
            )
            self._chunks.clear()
            self._id_to_point.clear()
            self._point_to_id.clear()
            self._counter = 0
        except Exception:
            pass

    def health_check(self) -> bool:
        try:
            return self._client is not None
        except Exception:
            return False
