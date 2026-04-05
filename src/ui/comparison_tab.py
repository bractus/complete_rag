import streamlit as st
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.libraries.base import LibraryAdapter
from src.pipeline.steps import RetrievalResult
from src.pipeline import embedder, tokenizer, steps
from src.ui import pipeline_viz


def render(adapters: list[LibraryAdapter]) -> None:
    st.header("Compare Retrieval")

    if not adapters:
        st.warning("No libraries enabled. Enable at least one library in the sidebar.")
        return

    all_empty = all(a.get_chunk_count() == 0 for a in adapters)
    if all_empty:
        st.info("No data ingested yet — use the Ingest tab first.")
        return

    query_text = st.text_input("Enter a query", placeholder="e.g. How do I make a beurre blanc?")

    if st.button("Search", type="primary") and query_text.strip():
        _run_comparison(query_text.strip(), adapters)


def _run_comparison(query_text: str, adapters: list[LibraryAdapter]) -> None:
    # Embed and tokenize
    with st.spinner("Embedding query..."):
        t0 = time.time()
        query_embedding = embedder.encode_single(query_text)
        inference_ms = (time.time() - t0) * 1000
        tok_step = tokenizer.tokenize(query_text)
        emb_step = steps.EmbeddingStep(
            embedding=query_embedding,
            dimensions=len(query_embedding),
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            inference_ms=inference_ms,
        )

    # Retrieve in parallel
    results: dict[str, RetrievalResult] = {}
    lib_params: dict = st.session_state.get("lib_params", {})

    def retrieve_one(adapter: LibraryAdapter) -> tuple[str, RetrievalResult]:
        params = lib_params.get(adapter.name, {})
        top_k = params.get("top_k", 5)
        extra = {k: v for k, v in params.items() if k != "top_k"}
        r = adapter.retrieve(query_embedding, top_k=top_k, **extra)
        r.query_text = query_text
        return adapter.name, r

    with st.spinner("Querying all libraries..."):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(retrieve_one, a): a.name for a in adapters}
            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result

    # Render results side-by-side
    cols = st.columns(len(adapters))
    for col, adapter in zip(cols, adapters):
        result = results.get(adapter.name)
        with col:
            st.subheader(adapter.name)
            if result is None:
                st.error("No response")
                continue
            if result.error:
                st.error(result.error)
                continue
            st.caption(f"{result.latency_ms:.0f}ms · {len(result.scores)} results")
            for s in result.scores:
                st.markdown(f"**#{s.rank}** `{s.cosine_score:.3f}`")
                st.text(s.chunk_text[:150] + ("..." if len(s.chunk_text) > 150 else ""))
                st.divider()

            # Pipeline visualization
            if result.scores:
                trace = steps.build_trace(
                    library_name=adapter.name,
                    query=steps.Query(
                        text=query_text,
                        embedding=query_embedding,
                        tokens=tok_step.tokens,
                        token_ids=tok_step.token_ids,
                    ),
                    result=result,
                    tokenization=tok_step,
                    embedding=emb_step,
                )
                pipeline_viz.render_pipeline(trace)
