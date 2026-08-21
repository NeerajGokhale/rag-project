"""Tests for the embedding module."""

from __future__ import annotations

import pytest

from src.embeddings.base import EmbeddingProvider
from src.embeddings.factory import get_embedding_provider
from src.config import EmbeddingsConfig
from tests.conftest import MockEmbeddingProvider


class TestEmbeddingProviderInterface:
    """Test that mock provider correctly implements the interface."""

    def test_mock_returns_correct_count(
        self, mock_embedding_provider: MockEmbeddingProvider
    ) -> None:
        texts = ["hello", "world", "test"]
        embeddings = mock_embedding_provider.embed(texts)
        assert len(embeddings) == 3

    def test_mock_returns_correct_dimension(
        self, mock_embedding_provider: MockEmbeddingProvider
    ) -> None:
        embeddings = mock_embedding_provider.embed(["hello"])
        assert len(embeddings[0]) == 64

    def test_mock_is_deterministic(
        self, mock_embedding_provider: MockEmbeddingProvider
    ) -> None:
        emb1 = mock_embedding_provider.embed(["hello"])
        emb2 = mock_embedding_provider.embed(["hello"])
        assert emb1 == emb2

    def test_different_texts_produce_different_embeddings(
        self, mock_embedding_provider: MockEmbeddingProvider
    ) -> None:
        embeddings = mock_embedding_provider.embed(["hello", "world"])
        assert embeddings[0] != embeddings[1]

    def test_empty_input(
        self, mock_embedding_provider: MockEmbeddingProvider
    ) -> None:
        embeddings = mock_embedding_provider.embed([])
        assert embeddings == []

    def test_mock_produces_unit_vectors(
        self, mock_embedding_provider: MockEmbeddingProvider
    ) -> None:
        embeddings = mock_embedding_provider.embed(["test"])
        magnitude = sum(v * v for v in embeddings[0]) ** 0.5
        assert abs(magnitude - 1.0) < 1e-5


class TestEmbeddingFactory:
    def test_unknown_provider_raises(self) -> None:
        config = EmbeddingsConfig(provider="unknown_provider", model="test")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider(config)
