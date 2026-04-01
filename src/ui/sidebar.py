import streamlit as st

from src.libraries.base import LibraryAdapter

# Parameters that are only meaningful for specific libraries
_LIBRARY_PARAMS = {
    "ChromaDB": [],
    "FAISS": [],
    "Qdrant": [
        {
            "key": "score_threshold",
            "label": "Score threshold",
            "type": "slider",
            "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05,
            "help": "Minimum cosine score required to include a result (0 = no filter)",
        }
    ],
    "LanceDB": [
        {
            "key": "metric",
            "label": "Distance metric",
            "type": "select",
            "options": ["cosine", "l2", "dot"],
            "default": "cosine",
            "help": "Distance metric used during vector search",
        }
    ],
    "Weaviate": [
        {
            "key": "certainty_threshold",
            "label": "Certainty threshold",
            "type": "slider",
            "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05,
            "help": "Minimum certainty score required to include a result (0 = no filter)",
        }
    ],
}


def render(all_adapters: dict[str, LibraryAdapter]) -> list[LibraryAdapter]:
    """Render the sidebar, collect per-library parameters, and return enabled adapters."""
    st.sidebar.title("Libraries")
    enabled = []
    lib_params: dict[str, dict] = {}

    for name, adapter in all_adapters.items():
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            checked = st.checkbox(name, value=True, key=f"sidebar_{name}")
        with col2:
            if not checked:
                st.write("—")
            elif adapter.health_check():
                st.write("✓")
            else:
                st.write("✗")

        with st.sidebar.expander("Parameters"):
            params: dict = {}

            params["top_k"] = st.slider(
                "Top K",
                min_value=1, max_value=10, value=5,
                key=f"param_{name}_top_k",
                help="Number of results to retrieve",
            )

            for spec in _LIBRARY_PARAMS.get(name, []):
                widget_key = f"param_{name}_{spec['key']}"
                if spec["type"] == "slider":
                    params[spec["key"]] = st.slider(
                        spec["label"],
                        min_value=spec["min"],
                        max_value=spec["max"],
                        value=spec["default"],
                        step=spec["step"],
                        key=widget_key,
                        help=spec["help"],
                    )
                elif spec["type"] == "select":
                    params[spec["key"]] = st.selectbox(
                        spec["label"],
                        options=spec["options"],
                        index=spec["options"].index(spec["default"]),
                        key=widget_key,
                        help=spec["help"],
                    )

        lib_params[name] = params

        if checked:
            enabled.append(adapter)

        st.sidebar.divider()

    st.session_state["lib_params"] = lib_params
    return enabled


def get_enabled_adapters(all_adapters: dict[str, LibraryAdapter]) -> list[LibraryAdapter]:
    return [
        adapter for name, adapter in all_adapters.items()
        if st.session_state.get(f"sidebar_{name}", True)
    ]
