"""ChromaDB vector store implementation."""

from __future__ import annotations

import logging

import chromadb

from src.retrieval.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB with persistent local storage.

    Uses deterministic IDs based on source_file::chunk_index to prevent
    duplicates when re-indexing.
    """

    def __init__(
        self,
        persist_dir: str = "data/processed/chroma_db",
        collection_name: str = "rag_documents",
    ) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB store ready (persist_dir=%s, collection=%s, count=%d)",
            persist_dir,
            collection_name,
            self._collection.count(),
        )

    @classmethod
    def in_memory(cls, collection_name: str = "rag_documents") -> "ChromaVectorStore":
        """Create an in-memory ChromaDB store (useful for testing).

        Args:
            collection_name: Name of the Chroma collection.

        Returns:
            A ChromaVectorStore backed by an ephemeral in-memory client.
        """
        instance = cls.__new__(cls)
        instance._client = chromadb.EphemeralClient()
        instance._collection = instance._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return instance

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Upsert chunks into ChromaDB.

        Args:
            ids: Deterministic IDs (e.g. "source_file::chunk_index").
            texts: The chunk texts.
            embeddings: The embedding vectors.
            metadatas: Metadata dicts for each chunk.
        """
        # ChromaDB metadata values must be str, int, float, or bool
        sanitized_metadatas = [
            {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
             for k, v in m.items()}
            for m in metadatas
        ]

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=sanitized_metadatas,
        )
        logger.info("Upserted %d chunk(s) into ChromaDB", len(ids))

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Query ChromaDB for similar chunks.

        ChromaDB returns *distances* for cosine space (distance = 1 - similarity),
        so we convert to similarity scores and filter by threshold.

        Args:
            embedding: The query embedding vector.
            top_k: Maximum number of results to return.
            score_threshold: Minimum cosine similarity score.

        Returns:
            A list of SearchResult objects, ordered by descending similarity.
        """
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []

        if not results["documents"] or not results["documents"][0]:
            return search_results

        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for doc, dist, meta in zip(documents, distances, metadatas):
            # Convert cosine distance to similarity
            score = 1.0 - dist
            if score >= score_threshold:
                search_results.append(
                    SearchResult(chunk_text=doc, score=score, metadata=meta or {})
                )

        return search_results
