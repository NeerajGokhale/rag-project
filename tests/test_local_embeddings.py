"""Tests for the local embedding provider (sentence-transformers)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import EmbeddingsConfig
from src.embeddings.factory import get_embedding_provider


# Create a mock sentence_transformers module so the lazy import works
_mock_st_module = MagicMock()
_mock_st_class = MagicMock()
_mock_st_module.SentenceTransformer = _mock_st_class


class TestLocalEmbeddingProvider:
    """Test LocalEmbeddingProvider with sentence_transformers mocked out."""

    def _make_provider(self):
        """Create a LocalEmbeddingProvider with sentence_transformers mocked."""
        from src.embeddings.local_embeddings import LocalEmbeddingProvider

        return LocalEmbeddingProvider(model="all-MiniLM-L6-v2")

    def test_embed_returns_correct_count(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            provider = self._make_provider()
            embeddings = provider.embed(["hello", "world", "test"])

        assert len(embeddings) == 3
        mock_model.encode.assert_called_once_with(
            ["hello", "world", "test"], convert_to_numpy=True
        )

    def test_embed_returns_lists_of_floats(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            provider = self._make_provider()
            embeddings = provider.embed(["hello"])

        assert isinstance(embeddings, list)
        assert isinstance(embeddings[0], list)
        assert all(isinstance(v, float) for v in embeddings[0])

    def test_embed_empty_input_returns_empty(self) -> None:
        mock_st = MagicMock()

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            provider = self._make_provider()
            embeddings = provider.embed([])

        assert embeddings == []
        # Model should not even be loaded for empty input
        mock_st.SentenceTransformer.assert_not_called()

    def test_model_loaded_lazily(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            provider = self._make_provider()
            # Model should not be loaded at construction time
            mock_st.SentenceTransformer.assert_not_called()

            provider.embed(["hello"])
            mock_st.SentenceTransformer.assert_called_once_with("all-MiniLM-L6-v2")

    def test_model_loaded_only_once(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            provider = self._make_provider()
            provider.embed(["hello"])
            provider.embed(["world"])

            # SentenceTransformer constructor called only once
            mock_st.SentenceTransformer.assert_called_once()


class TestEmbeddingFactoryLocal:
    """Test that the factory routes to LocalEmbeddingProvider."""

    def test_factory_returns_local_provider(self) -> None:
        from src.embeddings.local_embeddings import LocalEmbeddingProvider

        mock_st = MagicMock()
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            config = EmbeddingsConfig(provider="local", model="all-MiniLM-L6-v2", dimension=384)
            provider = get_embedding_provider(config)
            assert isinstance(provider, LocalEmbeddingProvider)
