"""Tests for vector store and retriever."""

from __future__ import annotations

import pytest

from src.retrieval.chroma_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import SearchResult
from tests.conftest import MockEmbeddingProvider


class TestChromaVectorStore:
    """Test ChromaDB vector store using in-memory client."""

    @pytest.fixture
    def store(self, request: pytest.FixtureRequest) -> ChromaVectorStore:
        # Use test name as collection name to isolate tests from each other
        return ChromaVectorStore.in_memory(collection_name=request.node.name)

    @pytest.fixture
    def mock_provider(self) -> MockEmbeddingProvider:
        return MockEmbeddingProvider(dimension=64)

    def test_upsert_and_query(
        self, store: ChromaVectorStore, mock_provider: MockEmbeddingProvider
    ) -> None:
        texts = ["machine learning is great", "deep learning uses neural networks"]
        embeddings = mock_provider.embed(texts)
        metadatas = [
            {"source_file": "doc.txt", "chunk_index": 0},
            {"source_file": "doc.txt", "chunk_index": 1},
        ]
        ids = ["doc.txt::c0", "doc.txt::c1"]

        store.upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

        # Query with the embedding of the first text — should return it as top result
        query_emb = mock_provider.embed(["machine learning is great"])[0]
        results = store.query(embedding=query_emb, top_k=2, score_threshold=0.0)

        assert len(results) > 0
        assert results[0].chunk_text == "machine learning is great"
        assert results[0].score > 0.5

    def test_dedup_on_reindex(
        self, store: ChromaVectorStore, mock_provider: MockEmbeddingProvider
    ) -> None:
        """Re-upserting with the same IDs should not create duplicates."""
        texts = ["hello world"]
        embeddings = mock_provider.embed(texts)
        ids = ["test::c0"]
        metadatas = [{"source_file": "test.txt", "chunk_index": 0}]

        # Upsert twice
        store.upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
        store.upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

        # Query should still return only one result
        query_emb = mock_provider.embed(["hello world"])[0]
        results = store.query(embedding=query_emb, top_k=10, score_threshold=0.0)
        assert len(results) == 1

    def test_score_threshold_filtering(
        self, store: ChromaVectorStore, mock_provider: MockEmbeddingProvider
    ) -> None:
        texts = ["cats are cute", "quantum physics is complex"]
        embeddings = mock_provider.embed(texts)
        ids = ["a::c0", "b::c0"]
        metadatas = [
            {"source_file": "a.txt", "chunk_index": 0},
            {"source_file": "b.txt", "chunk_index": 0},
        ]
        store.upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

        # Query with very high threshold — should filter most results
        query_emb = mock_provider.embed(["cats"])[0]
        results = store.query(embedding=query_emb, top_k=10, score_threshold=0.99)
        # May return zero results or only very close matches
        for r in results:
            assert r.score >= 0.99

    def test_empty_store_returns_empty(self, store: ChromaVectorStore) -> None:
        results = store.query(
            embedding=[0.0] * 64, top_k=5, score_threshold=0.0
        )
        assert results == []


class TestRetriever:
    def test_retrieve_returns_results(self) -> None:
        mock_provider = MockEmbeddingProvider(dimension=64)
        store = ChromaVectorStore.in_memory(collection_name="test_retriever")

        texts = [
            "Python is a programming language",
            "JavaScript runs in the browser",
        ]
        embeddings = mock_provider.embed(texts)
        ids = ["a::c0", "b::c0"]
        metadatas = [
            {"source_file": "python.txt", "chunk_index": 0},
            {"source_file": "js.txt", "chunk_index": 0},
        ]
        store.upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

        retriever = Retriever(
            embedding_provider=mock_provider, vector_store=store
        )
        results = retriever.retrieve(
            query="Python is a programming language",
            top_k=2,
            score_threshold=0.0,
        )

        assert len(results) > 0
        # The top result should be the Python chunk (exact match query)
        assert results[0].chunk_text == "Python is a programming language"

    def test_retrieve_respects_top_k(self) -> None:
        mock_provider = MockEmbeddingProvider(dimension=64)
        store = ChromaVectorStore.in_memory(collection_name="test_topk")

        texts = [f"document number {i}" for i in range(10)]
        embeddings = mock_provider.embed(texts)
        ids = [f"doc::c{i}" for i in range(10)]
        metadatas = [{"source_file": "doc.txt", "chunk_index": i} for i in range(10)]
        store.upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

        retriever = Retriever(
            embedding_provider=mock_provider, vector_store=store
        )
        results = retriever.retrieve(
            query="document number 0", top_k=3, score_threshold=0.0
        )

        assert len(results) <= 3
