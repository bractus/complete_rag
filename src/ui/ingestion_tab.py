import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.ingestion.document_loader import load_document
from src.ingestion.chunker import chunk_document
from src.pipeline.embedder import encode
from src.pipeline.steps import EmbeddedChunk
from src.libraries.base import LibraryAdapter, IngestionResult


def render(adapters: list[LibraryAdapter]) -> None:
    st.header("Ingest Documents")

    if not adapters:
        st.warning("No libraries enabled. Enable at least one library in the sidebar.")
        return

    uploaded_files = st.file_uploader(
        "Upload .txt or .md files",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more .txt or .md files to get started.")
        return

    if st.button("Ingest", type="primary"):
        all_chunks: list[EmbeddedChunk] = []
        errors = []

        with st.spinner("Loading and chunking documents..."):
            for f in uploaded_files:
                try:
                    doc = load_document(f)
                    chunks = chunk_document(doc)
                    texts = [c.text for c in chunks]
                    embeddings = encode(texts)
                    for chunk, emb in zip(chunks, embeddings):
                        all_chunks.append(EmbeddedChunk(
                            id=chunk.id,
                            source_filename=chunk.source_filename,
                            chunk_index=chunk.chunk_index,
                            text=chunk.text,
                            embedding=emb,
                        ))
                except Exception as e:
                    errors.append(f"**{f.name}**: {e}")

        if errors:
            for err in errors:
                st.error(err)

        if not all_chunks:
            st.warning("No chunks to ingest.")
            return

        st.info(f"Prepared {len(all_chunks)} chunks from {len(uploaded_files) - len(errors)} file(s). Ingesting into libraries...")

        results: dict[str, IngestionResult] = {}

        def ingest_one(adapter: LibraryAdapter) -> tuple[str, IngestionResult]:
            return adapter.name, adapter.ingest(all_chunks)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(ingest_one, a): a.name for a in adapters}
            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result

        # Status table
        st.subheader("Ingestion Results")
        for adapter in adapters:
            r = results.get(adapter.name)
            if r is None:
                continue
            if r.error:
                st.error(f"**{r.library_name}**: Failed — {r.error}")
            else:
                chunk_count = adapter.get_chunk_count()
                st.success(f"**{r.library_name}**: {r.chunks_stored} new chunks added (total: {chunk_count})")
