"""OpenAI embedding provider."""

from __future__ import annotations

import logging
import os

from openai import OpenAI

from src.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Wraps the OpenAI embeddings API.

    Reads the API key from the OPENAI_API_KEY environment variable (loaded
    by src.config via python-dotenv).
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Copy .env.example to .env and fill in your key."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        logger.info("Initialized OpenAI embedding provider (model=%s)", model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the OpenAI API.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors.
        """
        if not texts:
            return []

        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )

        # Sort by index to guarantee order matches input
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in sorted_data]
