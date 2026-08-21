"""Local embedding provider using sentence-transformers.

Runs entirely on CPU — no API key or paid service required.
The model is loaded lazily on the first call to ``embed()``.
"""

from __future__ import annotations

import logging

from src.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Wraps a sentence-transformers model for local embedding.

    Default model is ``all-MiniLM-L6-v2`` (384-dimensional, ~80 MB).
    No API key or network access is needed after the initial model download.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model
        self._model = None  # lazy-loaded
        logger.info("Initialized local embedding provider (model=%s)", model)

    def _load_model(self):
        """Load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded sentence-transformers model: %s", self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using a local sentence-transformers model.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors.
        """
        if not texts:
            return []

        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]
