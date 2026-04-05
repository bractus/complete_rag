import time
import numpy as np

from src.libraries.base import LibraryAdapter, IngestionResult, AdapterInitError
from src.pipeline.steps import EmbeddedChunk, RetrievalResult, SimilarityScore

COLLECTION_NAME = "RagChunk"


class WeaviateAdapter(LibraryAdapter):
    def __init__(self):
        self._client = None
        self._chunks: dict[str, EmbeddedChunk] = {}

    @property
    def name(self) -> str:
        return "Weaviate"

    def initialize(self) -> None:
        try:
            import weaviate
            self._client = weaviate.connect_to_embedded()
            # Create collection if it doesn't exist
            if not self._client.collections.exists(COLLECTION_NAME):
                import weaviate.classes.config as wc
                self._client.collections.create(
                    name=COLLECTION_NAME,
                    vectorizer_config=wc.Configure.Vectorizer.none(),
                    properties=[
                        wc.Property(name="chunk_id", data_type=wc.DataType.TEXT),
                        wc.Property(name="text", data_type=wc.DataType.TEXT),
                    ],
                )
        except Exception as e:
            raise AdapterInitError(f"Weaviate initialization failed: {e}")

    def ingest(self, chunks: list[EmbeddedChunk]) -> IngestionResult:
        try:
            collection = self._client.collections.get(COLLECTION_NAME)
            with collection.batch.dynamic() as batch:
                for c in chunks:
                    batch.add_object(
                        properties={"chunk_id": c.id, "text": c.text},
                        vector=c.embedding,
                    )
                    self._chunks[c.id] = c
            return IngestionResult(library_name=self.name, chunks_stored=len(chunks))
        except Exception as e:
            return IngestionResult(library_name=self.name, chunks_stored=0, error=str(e))

    def retrieve(self, query_embedding: list[float], top_k: int = 5, **kwargs) -> RetrievalResult:
        start = time.time()
        try:
            certainty_threshold = kwargs.get("certainty_threshold", 0.0)
            collection = self._client.collections.get(COLLECTION_NAME)
            results = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k,
                certainty=certainty_threshold if certainty_threshold > 0.0 else None,
                return_metadata=["certainty", "distance"],
            )
            scores = []
            for i, obj in enumerate(results.objects):
                cid = obj.properties.get("chunk_id", "")
                text = obj.properties.get("text", "")
                emb = self._chunks[cid].embedding if cid in self._chunks else []
                certainty = obj.metadata.certainty if obj.metadata and obj.metadata.certainty else 0.0
                cosine = 2.0 * certainty - 1.0 if certainty else 0.0
                scores.append(SimilarityScore(
                    chunk_id=cid,
                    chunk_text=text,
                    chunk_embedding=emb,
                    cosine_score=cosine,
                    native_score=certainty,
                    rank=i + 1,
                ))
            latency = (time.time() - start) * 1000
            return RetrievalResult(library_name=self.name, query_text="", scores=scores, latency_ms=latency)
        except Exception as e:
            return RetrievalResult(library_name=self.name, query_text="", scores=[], latency_ms=0, error=str(e))

    def get_chunk_count(self) -> int:
        try:
            collection = self._client.collections.get(COLLECTION_NAME)
            agg = collection.aggregate.over_all(total_count=True)
            return agg.total_count or 0
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            self._client.collections.delete(COLLECTION_NAME)
            import weaviate.classes.config as wc
            self._client.collections.create(
                name=COLLECTION_NAME,
                vectorizer_config=wc.Configure.Vectorizer.none(),
                properties=[
                    wc.Property(name="chunk_id", data_type=wc.DataType.TEXT),
                    wc.Property(name="text", data_type=wc.DataType.TEXT),
                ],
            )
            self._chunks.clear()
        except Exception:
            pass

    def health_check(self) -> bool:
        try:
            return self._client is not None and self._client.is_ready()
        except Exception:
            return False
