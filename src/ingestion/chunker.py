from src.pipeline.steps import Chunk, Document

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_document(document: Document, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    text = document.content
    chunks = []
    start = 0
    index = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                id=f"{document.filename}::{index}",
                source_filename=document.filename,
                chunk_index=index,
                text=chunk_text,
            ))
            index += 1
        start = end - overlap
        if start >= len(text):
            break
    return chunks
