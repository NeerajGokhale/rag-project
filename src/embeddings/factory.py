"""Factory function for embedding providers.

Returns the correct EmbeddingProvider based on the config, making it easy to
swap providers without touching any other code.
"""

from __future__ import annotations

from src.config import EmbeddingsConfig
from src.embeddings.base import EmbeddingProvider


def get_embedding_provider(config: EmbeddingsConfig) -> EmbeddingProvider:
    """Create an EmbeddingProvider from configuration.

    Args:
        config: The embeddings section of the application config.

    Returns:
        An initialized EmbeddingProvider.

    Raises:
        ValueError: If the configured provider is not recognized.
    """
    if config.provider == "openai":
        from src.embeddings.openai_embeddings import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(model=config.model)
    else:
        raise ValueError(
            f"Unknown embedding provider '{config.provider}'. "
            f"Supported providers: openai"
        )
