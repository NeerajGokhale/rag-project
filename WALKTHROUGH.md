# Step-by-Step Walkthrough

A beginner-friendly guide to getting the RAG pipeline up and running from scratch. Follow these steps in order the first time; afterwards you only need the [Quick Start](README.md#quick-start).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install Ollama](#2-install-ollama)
3. [Pull the LLM Model](#3-pull-the-llm-model)
4. [Create a Virtual Environment](#4-create-a-virtual-environment)
5. [Install Python Dependencies](#5-install-python-dependencies)
6. [Add Your Documents](#6-add-your-documents)
7. [Index Your Documents](#7-index-your-documents)
8. [Query the Pipeline](#8-query-the-pipeline)
9. [Switching to OpenAI Mode](#9-switching-to-openai-mode)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Requirement | Why |
|---|---|
| **macOS** (or Linux) | Ollama runs natively on macOS and Linux |
| **Python 3.11+** | Required by several dependencies |
| **Homebrew** (macOS) | Easiest way to install Ollama (`brew install ollama`) |
| **~3 GB free disk** | For the Ollama model (~2 GB) + embedding model (~80 MB) + ChromaDB data |

Check your Python version:

```bash
python3 --version
# Python 3.11.x or higher ✓
```

---

## 2. Install Ollama

Ollama lets you run LLMs locally. Install it with Homebrew:

```bash
brew install ollama
```

Or download the app directly from [ollama.com](https://ollama.com).

Verify the installation:

```bash
ollama --version
```

---

## 3. Pull the LLM Model

Download the `llama3.2` model (this is a one-time ~2 GB download):

```bash
ollama pull llama3.2
```

> **Tip:** You don't need to manually run `ollama serve` — the `./rag` wrapper script starts it automatically if it isn't already running. If you installed Ollama via the macOS app, it may also be running in the background already.

---

## 4. Create a Virtual Environment

Navigate to the project directory and create a virtual environment:

```bash
cd /path/to/rag-project
python3 -m venv venv
source venv/bin/activate
```

Your prompt should now show `(venv)`. (The `./rag` wrapper uses this venv automatically, so you only need to activate it manually for installing dependencies or running without the wrapper.)

---

## 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs everything you need, including:

- `sentence-transformers` — local embedding model (downloaded automatically on first use, ~80 MB)
- `requests` — HTTP calls to the Ollama API
- `chromadb` — vector database
- `pymupdf` — PDF parsing
- `tiktoken` — token-aware chunking

No API key or `.env` file is needed for local mode.

---

## 6. Add Your Documents

Place any `.txt`, `.md`, or `.pdf` files you want to query into the `data/raw/` directory:

```bash
cp ~/Documents/my-notes.txt data/raw/
cp ~/Downloads/paper.pdf data/raw/
```

The project ships with sample files in `data/raw/` so you can try it immediately.

**Supported formats:**

| Extension | Loader |
|---|---|
| `.txt` | Plain text |
| `.md` | Markdown (treated as plain text) |
| `.pdf` | Parsed with PyMuPDF, page numbers preserved |

---

## 7. Index Your Documents

Run the indexing command to chunk, embed, and store your documents:

```bash
./rag index
```

> **Note:** `./rag` is a bash script (macOS/Linux only). It checks if Ollama is running (starts it if not) and runs `main.py` using the project's virtual environment — no manual `source venv/bin/activate` or `ollama serve` needed. **Windows users:** use `python main.py index` directly instead (with the venv activated and Ollama running).

**Example output:**

```
09:12:34 [INFO] src.pipeline: Loading documents from data/raw …
09:12:34 [INFO] src.ingestion.loaders: Loaded sample.txt (2565 bytes)
09:12:34 [INFO] src.ingestion.loaders: Loaded sample2.txt (2310 bytes)
09:12:34 [INFO] src.pipeline: Chunking 2 document(s) …
09:12:35 [INFO] src.pipeline: Embedding 12 chunk(s) …
09:12:37 [INFO] src.pipeline: Storing in ChromaDB …

✅ Indexed 12 chunk(s) from data/raw
```

The first run takes a few extra seconds while the embedding model downloads. Subsequent runs are faster.

---

## 8. Query the Pipeline

Ask a question about your indexed documents:

```bash
./rag query "What is retrieval-augmented generation?"
```

**Example output:**

```
📝 Answer:
Retrieval-Augmented Generation (RAG) is a technique that enhances large
language models by retrieving relevant documents from a knowledge base
and including them as context in the prompt. This grounds the model's
responses in actual source material, reducing hallucination and allowing
the model to answer questions about private or domain-specific data.

📚 Sources:
  - sample.txt, chunk 0
  - sample.txt, chunk 2
  - sample2.txt, chunk 1
```

**Try debug mode** for more detail:

```bash
./rag --debug query "What is retrieval-augmented generation?"
```

This logs the full assembled prompt that gets sent to the LLM.

---

## 9. Switching to OpenAI Mode

If you want to use OpenAI models instead of (or alongside) local ones:

### Step 1 — Set your API key

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your real key:

```
OPENAI_API_KEY=sk-your-real-key-here
```

### Step 2 — Update `config.yaml`

Change one or both providers to `"openai"`:

```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"
  dimension: 1536

generation:
  provider: "openai"
  model: "gpt-4o-mini"
```

> **Note:** If you change the embedding provider, you must **re-index** (`./rag index`) because the vector dimensions will differ. Switching only the generation provider does not require re-indexing.

### Step 3 — Re-index (if you changed the embedding provider)

```bash
./rag index
```

You can also mix providers — for example, local embeddings (free) with OpenAI generation (for higher-quality answers). Just set each provider independently in `config.yaml`.

---

## 10. Troubleshooting

### Ollama not running

**Symptom:** `ConnectionRefusedError` or `Connection error` when running `query` or `index`.

**Fix:** If you're using `./rag`, this shouldn't happen — the wrapper starts Ollama automatically. If you're running `python main.py` directly, make sure Ollama is running:

```bash
# Check if Ollama is reachable
curl http://localhost:11434

# If not, start it
ollama serve
```

If you installed the Ollama macOS app, you can also launch it from Applications.

---

### Model not found

**Symptom:** Error mentioning the model isn't available.

**Fix:** Pull the model first:

```bash
ollama pull llama3.2
```

Verify it's downloaded:

```bash
ollama list
```

---

### Missing Python dependencies

**Symptom:** `ModuleNotFoundError` for `sentence_transformers`, `chromadb`, etc.

**Fix:** Make sure your virtual environment is activated and dependencies are installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

### Empty query results / "I don't have enough context"

**Symptom:** The LLM says it can't find relevant information.

**Possible causes:**

1. **You haven't indexed yet.** Run `./rag index` first.
2. **No relevant documents.** Make sure `data/raw/` contains files that relate to your question.
3. **Score threshold too high.** Lower `retrieval.score_threshold` in `config.yaml` (e.g. from `0.5` to `0.3`) to include more results.
4. **Wrong embedding provider after switching.** If you changed `embeddings.provider`, you need to re-index.

---

### Slow first run

The first time you run the pipeline, it downloads:

- The `all-MiniLM-L6-v2` embedding model (~80 MB)
- Any tokenizer data needed by `tiktoken`

This is a one-time cost. Subsequent runs start much faster.

---

### ChromaDB dimension mismatch

**Symptom:** Error about embedding dimension mismatch.

**Fix:** If you switched between local (384-dim) and OpenAI (1536-dim) embeddings, delete the old database and re-index:

```bash
rm -rf data/processed/chroma_db
./rag index
```
