import pytest
from io import BytesIO
from unittest.mock import MagicMock

from src.ingestion.document_loader import load_document
from src.pipeline.steps import Document


class MockUploadedFile:
    """Mock object that mimics Streamlit's UploadedFile."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def read(self) -> bytes:
        return self._content


def test_load_valid_txt_file():
    f = MockUploadedFile("sample.txt", b"Hello, this is some content.")
    doc = load_document(f)
    assert isinstance(doc, Document)
    assert doc.filename == "sample.txt"
    assert doc.content == "Hello, this is some content."
    assert doc.size_bytes == len(b"Hello, this is some content.")


def test_load_valid_md_file():
    content = b"# French Cuisine\n\nSome great recipes here."
    f = MockUploadedFile("cookbook.md", content)
    doc = load_document(f)
    assert doc.filename == "cookbook.md"
    assert "French Cuisine" in doc.content
    assert doc.size_bytes == len(content)


def test_invalid_extension_raises_value_error():
    f = MockUploadedFile("document.pdf", b"PDF content")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(f)


def test_invalid_extension_csv_raises_value_error():
    f = MockUploadedFile("data.csv", b"col1,col2\n1,2")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(f)


def test_empty_file_raises_value_error():
    f = MockUploadedFile("empty.txt", b"")
    with pytest.raises(ValueError, match="empty or contains only whitespace"):
        load_document(f)


def test_whitespace_only_file_raises_value_error():
    f = MockUploadedFile("whitespace.md", b"   \n\n\t  ")
    with pytest.raises(ValueError, match="empty or contains only whitespace"):
        load_document(f)


def test_extension_case_insensitive():
    f = MockUploadedFile("NOTES.TXT", b"Some notes here.")
    doc = load_document(f)
    assert doc.filename == "NOTES.TXT"
    assert doc.content == "Some notes here."


def test_document_upload_ts_is_set():
    f = MockUploadedFile("test.txt", b"Content here.")
    doc = load_document(f)
    assert doc.upload_ts is not None
