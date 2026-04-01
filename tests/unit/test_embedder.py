import pytest

from src.pipeline.embedder import encode, encode_single


def test_encode_single_returns_list_of_floats():
    result = encode_single("hello world")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(v, float) for v in result)


def test_encode_single_returns_384_dimensions():
    result = encode_single("What is a beurre blanc sauce?")
    assert len(result) == 384


def test_encode_batch_returns_two_vectors():
    results = encode(["first sentence", "second sentence"])
    assert len(results) == 2
    assert len(results[0]) == 384
    assert len(results[1]) == 384


def test_encode_batch_all_floats():
    results = encode(["hello", "world", "test"])
    for vec in results:
        assert all(isinstance(v, float) for v in vec)


def test_model_loads_without_error():
    # Simply calling encode_single verifies the model loads
    result = encode_single("test")
    assert result is not None


def test_encode_single_normalized():
    import math
    result = encode_single("normalize test")
    norm = math.sqrt(sum(v * v for v in result))
    # Should be close to 1.0 due to normalize_embeddings=True
    assert abs(norm - 1.0) < 1e-5
