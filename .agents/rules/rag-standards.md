# RAG Project Standards

## Stack
- Python 3.11+
- Type hints required on all function signatures
- Dependency management via requirements.txt (keep it pinned to major versions)

## Project structure
- src/ingestion/    -> load raw docs (pdf, txt, md, html) and clean/normalize text
- src/embeddings/   -> embedding model wrapper(s), single interface regardless of provider
- src/retrieval/    -> vector store client + retriever logic (top-k search, filtering)
- src/generation/   -> prompt templates + LLM call wrapper
- src/eval/         -> retrieval and answer-quality evaluation scripts
- data/raw/         -> original source documents, never modified
- data/processed/   -> chunked/cleaned intermediate data (gitignored)

## Chunking defaults
- Chunk size: 512 tokens
- Overlap: 50 tokens
- Always keep chunk provenance: {source_file, page_number/section, chunk_index}

## Embeddings & retrieval
- Every retrieval function returns a list of (chunk_text, score, metadata) tuples
- Default top_k = 5, configurable
- Include a similarity score threshold to filter weak matches (default 0.5, cosine)

## Generation
- Prompt templates live in src/generation/prompts.py as named constants, not inline strings
- Always inject retrieved chunks with clear delimiters and cite source metadata in the prompt
- Log the full assembled prompt for any query when running in debug mode

## Secrets & config
- No hardcoded API keys, ever — read from environment variables via python-dotenv
- All tunable parameters (chunk size, top_k, model name) go in a single config.yaml, not scattered across files

## Testing
- Any new module in src/ ships with a corresponding test in the mirrored path under tests/
- Retrieval changes must include at least one test asserting expected chunks are returned for a sample query
- Run `pytest` before considering any task complete

## Git hygiene
- Small, focused commits with clear messages
- Never commit data/processed/, .env, or model weight files
