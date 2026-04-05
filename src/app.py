import sys
import os

# Ensure project root is on sys.path so `src` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import streamlit as st

from src.libraries.base import AdapterInitError
from src.libraries.chromadb_store import ChromaDBAdapter
from src.libraries.faiss_store import FAISSAdapter
from src.libraries.qdrant_store import QdrantAdapter
from src.libraries.lancedb_store import LanceDBAdapter
from src.libraries.weaviate_store import WeaviateAdapter
from src.ui import sidebar, ingestion_tab, comparison_tab

st.set_page_config(page_title="RAG Library Comparison Explorer", layout="wide")


def _init_adapters():
    """Initialize all five adapters and store in session state."""
    if "adapters" not in st.session_state:
        # Clean up LanceDB temp dir on first load
        if os.path.exists(".lancedb_tmp"):
            shutil.rmtree(".lancedb_tmp", ignore_errors=True)

        adapter_classes = [
            ChromaDBAdapter,
            FAISSAdapter,
            QdrantAdapter,
            LanceDBAdapter,
            WeaviateAdapter,
        ]
        adapters = {}
        for cls in adapter_classes:
            a = cls()
            try:
                a.initialize()
                adapters[a.name] = a
            except AdapterInitError as e:
                # Create a stub adapter that always returns errors
                adapters[a.name] = a  # still store it; health_check will return False
                st.session_state[f"init_error_{a.name}"] = str(e)
        st.session_state["adapters"] = adapters


_init_adapters()

all_adapters = st.session_state["adapters"]

# Show any init errors
for name in all_adapters:
    err_key = f"init_error_{name}"
    if err_key in st.session_state:
        st.warning(f"**{name}** failed to initialize: {st.session_state[err_key]}")

# Sidebar: library selection
enabled_adapters = sidebar.render(all_adapters)

# Main tabs
tab_ingest, tab_compare = st.tabs(["Ingest", "Compare"])

with tab_ingest:
    ingestion_tab.render(enabled_adapters)

with tab_compare:
    comparison_tab.render(enabled_adapters)
