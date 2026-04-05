import pytest

from src.ingestion.chunker import chunk_document, CHUNK_SIZE, CHUNK_OVERLAP
from src.pipeline.steps import Chunk, Document


def make_document(content: str, filename: str = "test.txt") -> Document:
    return Document(filename=filename, content=content, size_bytes=len(content.encode()))


def test_short_text_produces_single_chunk():
    doc = make_document("Short text that fits in one chunk.")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text that fits in one chunk."


def test_long_text_produces_multiple_chunks():
    # Create text longer than CHUNK_SIZE
    content = "A" * 1200  # should produce at least 2 chunks
    doc = make_document(content)
    chunks = chunk_document(doc)
    assert len(chunks) > 1


def test_chunk_id_format():
    doc = make_document("Some content here", filename="myfile.txt")
    chunks = chunk_document(doc)
    assert chunks[0].id == "myfile.txt::0"


def test_chunk_ids_are_sequential():
    content = "X" * 1500  # large enough for multiple chunks
    doc = make_document(content, filename="seq.txt")
    chunks = chunk_document(doc)
    for i, chunk in enumerate(chunks):
        assert chunk.id == f"seq.txt::{i}"
        assert chunk.chunk_index == i
        assert chunk.source_filename == "seq.txt"


def test_overlap_is_respected():
    # With 500-char chunks and 50-char overlap, second chunk starts at index 450
    content = "A" * 500 + "B" * 500  # 1000 chars total
    doc = make_document(content)
    chunks = chunk_document(doc)
    # First chunk: chars 0-499 -> all A's
    # Second chunk: starts at 450 (500-50), so chars 450-949 -> 50 A's + 500 B's
    assert len(chunks) >= 2
    first = chunks[0].text
    second = chunks[1].text
    # There should be some overlap: first chunk ends with A's, second chunk starts with A's
    assert first[-1] == "A"
    assert second[0] == "A"
    # The beginning of the second chunk overlaps with the end of the first
    overlap_region = second[:CHUNK_OVERLAP]
    assert all(c == "A" for c in overlap_region)


def test_empty_content_produces_no_chunks():
    # Document with just whitespace-strippable content in chunks
    content = " " * 600  # all spaces; chunk_text.strip() will be empty
    doc = make_document(content)
    chunks = chunk_document(doc)
    assert chunks == []


def test_chunk_text_is_stripped():
    # Leading/trailing whitespace in chunk text should be stripped
    content = "  Hello world  " + " " * 485 + "  More text  "
    doc = make_document(content)
    chunks = chunk_document(doc)
    for chunk in chunks:
        assert chunk.text == chunk.text.strip()


def test_custom_chunk_size():
    content = "A" * 200
    doc = make_document(content)
    chunks = chunk_document(doc, chunk_size=100, overlap=10)
    # With 200 chars and 100-char chunks (10 overlap), expect ~2 chunks
    assert len(chunks) >= 2
    assert len(chunks[0].text) <= 100
