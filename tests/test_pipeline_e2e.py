"""End-to-end pipeline test.

Ingests a sample document → indexes with mock embeddings → queries →
asserts that the relevant chunk is retrieved. No API keys required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config, ChunkingConfig, EmbeddingsConfig, GenerationConfig, RetrievalConfig, VectorStoreConfig
from src.ingestion.loaders import load_directory
from src.ingestion.chunker import chunk_documents
from src.retrieval.chroma_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from tests.conftest import MockEmbeddingProvider


class TestEndToEndPipeline:
    """End-to-end test: ingest → chunk → embed → index → retrieve."""

    @pytest.fixture
    def e2e_data_dir(self, tmp_path: Path) -> Path:
        """Create a directory with a sample document."""
        data_dir = tmp_path / "raw"
        data_dir.mkdir()

        content = (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances "
            "large language model responses by retrieving relevant information "
            "from an external knowledge base before generating an answer. "
            "Instead of relying solely on the model's training data, RAG first "
            "searches a document store to find passages related to the user's query. "
            "The key advantage of RAG is reduced hallucination since answers are "
            "grounded in retrieved evidence. RAG systems can be evaluated using "
            "metrics such as retrieval recall and answer faithfulness."
        )
        (data_dir / "rag_overview.txt").write_text(content, encoding="utf-8")
        return data_dir

    def test_ingest_and_retrieve(self, e2e_data_dir: Path) -> None:
        """Full pipeline: ingest a doc, index it, and query for relevant chunks."""
        mock_provider = MockEmbeddingProvider(dimension=64)
        store = ChromaVectorStore.in_memory(collection_name="e2e_test")

        # 1. Load documents
        documents = load_directory(e2e_data_dir)
        assert len(documents) >= 1

        # 2. Chunk documents (small chunk size for testing)
        chunks = chunk_documents(documents, chunk_size=50, overlap=10)
        assert len(chunks) >= 1

        # 3. Embed
        texts = [c.text for c in chunks]
        embeddings = mock_provider.embed(texts)
        assert len(embeddings) == len(chunks)

        # 4. Upsert
        ids = [f"{c.metadata['source_file']}::c{c.metadata['chunk_index']}" for c in chunks]
        metadatas = [c.metadata for c in chunks]
        store.upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

        # 5. Retrieve
        retriever = Retriever(
            embedding_provider=mock_provider,
            vector_store=store,
        )

        # Query with the exact text of a chunk — should retrieve it
        results = retriever.retrieve(
            query=texts[0],
            top_k=3,
            score_threshold=0.0,
        )

        assert len(results) > 0
        # The top result should be an exact or near-exact match
        assert results[0].score > 0.5
        # Metadata should be preserved
        assert "source_file" in results[0].metadata

    def test_reindex_does_not_duplicate(self, e2e_data_dir: Path) -> None:
        """Re-indexing with the same data should not create duplicates."""
        mock_provider = MockEmbeddingProvider(dimension=64)
        store = ChromaVectorStore.in_memory(collection_name="e2e_reindex")

        documents = load_directory(e2e_data_dir)
        chunks = chunk_documents(documents, chunk_size=50, overlap=10)
        texts = [c.text for c in chunks]
        embeddings = mock_provider.embed(texts)
        ids = [f"{c.metadata['source_file']}::c{c.metadata['chunk_index']}" for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Index twice
        store.upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
        store.upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

        # Query — should not get duplicates
        query_emb = mock_provider.embed([texts[0]])[0]
        results = store.query(embedding=query_emb, top_k=20, score_threshold=0.0)

        # Count results with the same chunk text
        matching = [r for r in results if r.chunk_text == texts[0]]
        assert len(matching) == 1, "Re-indexing produced duplicates"
