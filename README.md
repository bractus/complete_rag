# RAG Library Comparison Explorer

A local Streamlit application that ingests text documents into five vector store libraries simultaneously and lets you compare their retrieval results side by side. For each query, the app visualizes every step of the retrieval pipeline — tokenization, embedding, cosine similarity, and ranked results — so you can see exactly how each library behaves under the same conditions.

## Libraries compared

- ChromaDB
- FAISS
- Weaviate (embedded mode)
- Qdrant (in-memory)
- LanceDB

All five run locally. No cloud accounts or API keys are required for the comparison tool itself.

## How it works

1. You upload `.txt` or `.md` files. The app splits them into 500-character chunks and embeds each chunk using `sentence-transformers/all-MiniLM-L6-v2` — one shared model used across all libraries to keep comparisons fair.
2. The embedded chunks are stored in all five libraries at once.
3. You type a natural language query. The app queries all active libraries in parallel and displays the top results in side-by-side columns.
4. Expanding the **Pipeline Detail** panel under any library shows the full trace: the tokenized query, a visualization of the embedding vector, cosine similarity scores for each candidate chunk, and the final ranked list.

## Requirements

- Python 3.11 or later
- ~2 GB free disk space (model weights + Weaviate binary, downloaded once on first run)
- macOS or Linux (Windows via WSL is untested but should work)

## Setup

```bash
git clone <repo-url>
cd complete_rag

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

On first launch, `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~90 MB) and Weaviate downloads its embedded binary (~50 MB). Both are cached and only downloaded once.

## Running the app

```bash
source venv/bin/activate
streamlit run src/app.py
```

The app opens at `http://localhost:8501`.

## Usage

**Sidebar** — Check or uncheck each library to include or exclude it from ingestion and retrieval. All five are enabled by default.

**Ingest tab**

1. Click **Browse files** and select one or more `.txt` or `.md` files.
2. Click **Ingest**. The app chunks, embeds, and stores the content in all active libraries in parallel.
3. Each library reports the number of chunks stored and any errors.

**Compare tab**

1. Enter a natural language query and click **Search**.
2. Results from each active library appear side by side, each showing ranked chunks with cosine similarity scores.
3. Click **Pipeline Detail** under any library column to inspect the step-by-step trace.

Reloading the browser page starts a fresh session and clears all in-memory indexes.

## Project structure

```
src/
  app.py                  # Streamlit entry point
  ingestion/              # Document loading and chunking
  libraries/              # Adapter for each vector store
  pipeline/               # Shared embedding model, tokenizer, dataclasses
  ui/                     # Streamlit tab and sidebar components
tests/
  unit/                   # Unit tests for ingestion and pipeline
  integration/            # Round-trip ingest+retrieve tests per library
```

## Running tests

```bash
source venv/bin/activate
pytest tests/
```

Weaviate integration tests are skipped automatically if the embedded binary is unavailable.

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `ModuleNotFoundError: No module named 'src'` | Run `streamlit run src/app.py` from the project root, not from inside `src/` |
| Weaviate fails to start | Check available disk space; delete `~/.local/share/weaviate` and retry |
| FAISS import error | Confirm `faiss-cpu` is installed (not `faiss-gpu` unless you have a GPU) |
| Model download hangs | Check your internet connection; the model is cached in `~/.cache/huggingface/` after the first download |
| LanceDB permission error | Ensure the current directory is writable; a `.lancedb_tmp/` directory is created at runtime |

To clear LanceDB's temporary files manually:

```bash
rm -rf .lancedb_tmp/
```
