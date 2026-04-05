from sentence_transformers import SentenceTransformer
import numpy as np

_model = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def encode_single(text: str) -> list[float]:
    return encode([text])[0]
