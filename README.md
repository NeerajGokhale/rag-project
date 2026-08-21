# RAG Pipeline

A complete Retrieval-Augmented Generation pipeline in Python. Ingests documents, chunks them, embeds them, stores in ChromaDB, and generates grounded answers with source citations. **Runs fully free by default** using local embeddings (sentence-transformers) and a local LLM (Ollama) — no API keys required.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Index documents
python main.py index

# 3. Query
python main.py query "What is retrieval-augmented generation?"
```

> **Note:** The default config uses local providers. If you want to use OpenAI instead, see [Switching to OpenAI](#switching-to-openai) below.

## Running Fully Free (Local Mode)

The default configuration runs entirely on your machine with no paid APIs:

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.com
```

### 2. Pull a model

```bash
ollama pull llama3.2
```

### 3. Start Ollama (if not already running)

```bash
ollama serve
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs `sentence-transformers` (for local embeddings) and `requests` (for Ollama API calls). The embedding model (`all-MiniLM-L6-v2`, ~80 MB) is downloaded automatically on first use.

### 5. Confirm config.yaml

Ensure both providers are set to `"local"` (this is the default):

```yaml
embeddings:
  provider: "local"
  model: "all-MiniLM-L6-v2"

generation:
  provider: "local"
  model: "llama3.2"
```

That's it — no `.env` file or API key needed!

## Switching to OpenAI

To use OpenAI instead:

1. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`
2. Update `config.yaml`:

```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"
  dimension: 1536

generation:
  provider: "openai"
  model: "gpt-4o-mini"
```

## Project Structure

```
src/
├── config.py              # Central config loader (config.yaml + .env)
├── pipeline.py            # End-to-end pipeline orchestrator
├── ingestion/
│   ├── loaders.py         # File loaders (txt, md, pdf)
│   └── chunker.py         # Token-aware chunking with overlap
├── embeddings/
│   ├── base.py            # Abstract EmbeddingProvider interface
│   ├── openai_embeddings.py  # OpenAI implementation
│   ├── local_embeddings.py   # Local implementation (sentence-transformers)
│   └── factory.py         # Provider factory
├── retrieval/
│   ├── vector_store.py    # Abstract VectorStore + SearchResult
│   ├── chroma_store.py    # ChromaDB implementation
│   └── retriever.py       # Query embedding + search
├── generation/
│   ├── prompts.py         # Prompt templates (named constants)
│   ├── llm.py             # OpenAI LLM client
│   ├── local_llm.py       # Local LLM client (Ollama)
│   └── factory.py         # LLM client factory
└── eval/                  # Evaluation scripts (future)

data/
├── raw/                   # Source documents (add yours here)
└── processed/             # Chunked data & ChromaDB storage (gitignored)

tests/                     # Mirrors src/ structure
config.yaml                # All tunable parameters
main.py                    # CLI entry point
```

## Configuration

All tunable parameters are in [`config.yaml`](config.yaml):

| Parameter | Default | Description |
|---|---|---|
| `embeddings.provider` | local | `"local"` (free) or `"openai"` (API key required) |
| `embeddings.model` | all-MiniLM-L6-v2 | Model name (varies by provider) |
| `embeddings.dimension` | 384 | Embedding dimensions (384 for local, 1536 for OpenAI) |
| `generation.provider` | local | `"local"` (Ollama, free) or `"openai"` (API key required) |
| `generation.model` | llama3.2 | Model name (varies by provider) |
| `generation.temperature` | 0.2 | LLM temperature |
| `generation.max_tokens` | 1024 | Max output tokens |
| `chunking.chunk_size` | 512 | Max tokens per chunk |
| `chunking.overlap` | 50 | Overlapping tokens between chunks |
| `retrieval.top_k` | 5 | Max results per query |
| `retrieval.score_threshold` | 0.5 | Min cosine similarity |

## Adding Documents

Drop `.txt`, `.md`, or `.pdf` files into `data/raw/`, then run:

```bash
python main.py index
```

Re-indexing is safe — duplicate chunks are upserted, not duplicated.

## Running Tests

```bash
pytest tests/ -v
```

All tests use mocks — no API key, Ollama server, or model download required.

## Debug Mode

```bash
python main.py --debug query "Your question"
```

Logs the full assembled prompt sent to the LLM.
