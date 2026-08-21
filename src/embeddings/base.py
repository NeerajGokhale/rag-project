"""Abstract base class for embedding providers.

Every embedding provider must implement the single `embed` method so the rest
of the pipeline is decoupled from any specific vendor or model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface that all embedding providers must implement."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """
        ...
