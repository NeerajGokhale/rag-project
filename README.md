# RAG Pipeline

A complete Retrieval-Augmented Generation pipeline in Python. Ingests documents, chunks them, embeds with OpenAI, stores in ChromaDB, and generates grounded answers with source citations.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Index documents
python main.py index

# 4. Query
python main.py query "What is retrieval-augmented generation?"
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
│   └── factory.py         # Provider factory
├── retrieval/
│   ├── vector_store.py    # Abstract VectorStore + SearchResult
│   ├── chroma_store.py    # ChromaDB implementation
│   └── retriever.py       # Query embedding + search
├── generation/
│   ├── prompts.py         # Prompt templates (named constants)
│   └── llm.py             # LLM client (OpenAI chat completions)
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
| `chunking.chunk_size` | 512 | Max tokens per chunk |
| `chunking.overlap` | 50 | Overlapping tokens between chunks |
| `embeddings.provider` | openai | Embedding provider |
| `embeddings.model` | text-embedding-3-small | Embedding model |
| `retrieval.top_k` | 5 | Max results per query |
| `retrieval.score_threshold` | 0.5 | Min cosine similarity |
| `generation.model` | gpt-4o-mini | LLM model |
| `generation.temperature` | 0.2 | LLM temperature |

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

All tests use mocks — no API key required.

## Debug Mode

```bash
python main.py --debug query "Your question"
```

Logs the full assembled prompt sent to the LLM.
