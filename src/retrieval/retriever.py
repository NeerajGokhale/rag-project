"""Retriever: embeds a query and searches the vector store."""

from __future__ import annotations

import logging

from src.embeddings.base import EmbeddingProvider
from src.retrieval.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Orchestrates query embedding and vector store search.

    Sits between the user query and the vector store, handling the embedding
    step and score-threshold filtering.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The user's natural-language question.
            top_k: Maximum number of results.
            score_threshold: Minimum cosine similarity score.

        Returns:
            A list of (chunk_text, score, metadata) results, highest score first.
        """
        logger.debug("Retrieving for query: %s", query[:100])

        # Embed the query
        query_embedding = self._embedding_provider.embed([query])[0]

        # Search the vector store
        results = self._vector_store.query(
            embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        logger.info(
            "Retrieved %d result(s) (top_k=%d, threshold=%.2f)",
            len(results),
            top_k,
            score_threshold,
        )
        return results
