import os

from src.pipeline.steps import Document

ALLOWED_EXTENSIONS = {".txt", ".md"}


def load_document(uploaded_file) -> Document:
    """uploaded_file: Streamlit UploadedFile or any object with .name and .read()"""
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Only .txt and .md are accepted.")
    content_bytes = uploaded_file.read()
    if isinstance(content_bytes, bytes):
        content = content_bytes.decode("utf-8", errors="replace")
    else:
        content = content_bytes
    content = content.strip()
    if not content:
        raise ValueError(f"File '{name}' is empty or contains only whitespace.")
    return Document(
        filename=name,
        content=content,
        size_bytes=len(content_bytes) if isinstance(content_bytes, (bytes, bytearray)) else len(content_bytes.encode("utf-8")),
    )
