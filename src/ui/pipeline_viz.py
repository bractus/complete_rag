import streamlit as st
import plotly.express as px
import pandas as pd

from src.pipeline.steps import PipelineTrace


def render_pipeline(trace: PipelineTrace) -> None:
    with st.expander("Pipeline Detail", expanded=False):
        # Step 1: Tokenization
        st.markdown("**Step 1 — Tokenization**")
        tokens = trace.step_tokenization.tokens
        st.code(" | ".join(tokens) if tokens else "(no tokens)", language=None)
        st.caption(f"{len(tokens)} tokens · vocabulary size: {trace.step_tokenization.vocabulary_size:,}")

        st.divider()

        # Step 2: Embedding
        st.markdown("**Step 2 — Embedding**")
        emb = trace.step_embedding.embedding
        st.caption(
            f"{trace.step_embedding.dimensions}-dimensional vector · "
            f"model: `{trace.step_embedding.model_name}` · "
            f"inference: {trace.step_embedding.inference_ms:.1f}ms"
        )
        if emb:
            display_dims = min(50, len(emb))
            fig = px.bar(
                x=list(range(display_dims)),
                y=emb[:display_dims],
                labels={"x": "Dimension", "y": "Value"},
                title=f"First {display_dims} dimensions",
                height=200,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key=f"emb_chart_{trace.library_name}")

        st.divider()

        # Step 3: Cosine Similarity Scores
        st.markdown("**Step 3 — Cosine Similarity**")
        scores = trace.step_similarity.candidate_scores
        if scores:
            df = pd.DataFrame([
                {
                    "Rank": s.rank,
                    "Cosine Score": f"{s.cosine_score:.4f}",
                    "Chunk Preview": s.chunk_text[:80] + ("..." if len(s.chunk_text) > 80 else ""),
                }
                for s in scores
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No scores available.")

        st.divider()

        # Step 4: Ranked Results
        st.markdown("**Step 4 — Ranked Results**")
        for s in trace.step_ranking.scores:
            st.markdown(f"**#{s.rank}** — score: `{s.cosine_score:.4f}`")
            st.text(s.chunk_text[:200] + ("..." if len(s.chunk_text) > 200 else ""))
