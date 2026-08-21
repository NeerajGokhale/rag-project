# build-rag-pipeline

Build or extend the RAG pipeline in this repository, following .agents/rules/rag-standards.md.

Steps:
1. Ingestion
   - Implement/update loaders in src/ingestion/ for files in data/raw
   - Support at minimum: .txt, .md, .pdf
   - Output cleaned text with source metadata preserved

2. Chunking
   - Split cleaned text into overlapping chunks per the defaults in rag-standards.md
   - Store chunk_index and source metadata alongside each chunk

3. Embeddings
   - Implement src/embeddings/ wrapper with a single embed(texts: list[str]) -> list[vector] interface
   - Make the embedding provider/model swappable via config.yaml

4. Vector store
   - Implement src/retrieval/ to upsert (chunk_text, embedding, metadata) into a vector store
   - Default to a local/self-hosted option (e.g. Chroma) unless config.yaml specifies otherwise
   - Support re-indexing without duplicating existing chunks

5. Retrieval
   - Implement top-k semantic search with the score threshold from config.yaml
   - Return results as (chunk_text, score, metadata)

6. Generation
   - Build the prompt template in src/generation/prompts.py that injects retrieved chunks + user query
   - Call the configured LLM and return the answer plus the source metadata used

7. Tests
   - Add/update tests in tests/ mirroring any new or changed module
   - Include at least one end-to-end test: ingest a sample doc -> query -> assert relevant chunk retrieved

8. Verification
   - Run `pytest` and report pass/fail
   - Summarize what was built/changed and any open decisions (e.g. vector store choice, embedding model) for my review
