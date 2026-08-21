"""Abstract base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    chunk_text: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    """Interface that all vector store backends must implement."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Insert or update chunks in the store.

        Using deterministic IDs enables re-indexing without duplicates.

        Args:
            ids: Unique identifier for each chunk.
            texts: The chunk texts.
            embeddings: The embedding vectors.
            metadatas: Metadata dicts for each chunk.
        """
        ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Query the store for similar chunks.

        Args:
            embedding: The query embedding vector.
            top_k: Maximum number of results to return.
            score_threshold: Minimum cosine similarity score.

        Returns:
            A list of SearchResult objects, ordered by descending similarity.
        """
        ...
