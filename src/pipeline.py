"""RAG Pipeline orchestrator.

Wires together ingestion → chunking → embedding → vector-store upsert →
retrieval → generation into a single, easy-to-use interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.config import Config, load_config
from src.embeddings.base import EmbeddingProvider
from src.embeddings.factory import get_embedding_provider
from src.generation.factory import get_llm_client
from src.generation.llm import GenerationResult
from src.ingestion.chunker import Chunk, chunk_documents
from src.ingestion.loaders import Document, load_directory
from src.retrieval.chroma_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _make_chunk_id(chunk: Chunk) -> str:
    """Create a deterministic ID for a chunk to prevent duplicates."""
    source = chunk.metadata.get("source_file", "unknown")
    page = chunk.metadata.get("page_number", "")
    idx = chunk.metadata.get("chunk_index", 0)
    if page:
        return f"{source}::p{page}::c{idx}"
    return f"{source}::c{idx}"


class RAGPipeline:
    """End-to-end RAG pipeline.

    Usage::

        pipeline = RAGPipeline()
        pipeline.index("data/raw")
        result = pipeline.query("What is retrieval-augmented generation?")
        print(result.answer)
    """

    def __init__(
        self,
        config: Config | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        llm_client=None,
    ) -> None:
        self._config = config or load_config()

        # Components — injectable for testing
        self._embedding_provider = embedding_provider or get_embedding_provider(
            self._config.embeddings
        )
        self._vector_store = vector_store or ChromaVectorStore(
            persist_dir=self._config.vector_store.persist_dir,
            collection_name=self._config.vector_store.collection_name,
        )
        self._llm_client = llm_client or get_llm_client(self._config.generation)
        self._retriever = Retriever(
            embedding_provider=self._embedding_provider,
            vector_store=self._vector_store,
        )

    def index(self, data_dir: str | Path) -> int:
        """Ingest, chunk, embed, and upsert all documents from a directory.

        Args:
            data_dir: Path to the directory containing source documents.

        Returns:
            The number of chunks indexed.
        """
        data_path = Path(data_dir)
        logger.info("Indexing documents from %s", data_path)

        # 1. Load documents
        documents: list[Document] = load_directory(data_path)
        if not documents:
            logger.warning("No documents found in %s", data_path)
            return 0

        # 2. Chunk documents
        chunks: list[Chunk] = chunk_documents(
            documents,
            chunk_size=self._config.chunking.chunk_size,
            overlap=self._config.chunking.overlap,
        )
        if not chunks:
            logger.warning("Chunking produced zero chunks")
            return 0

        # 3. Embed all chunks
        texts = [c.text for c in chunks]
        logger.info("Embedding %d chunk(s)...", len(texts))
        embeddings = self._embedding_provider.embed(texts)

        # 4. Upsert into vector store
        ids = [_make_chunk_id(c) for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self._vector_store.upsert(
            ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas
        )

        logger.info("Indexed %d chunk(s) from %d document(s)", len(chunks), len(documents))
        return len(chunks)

    def query(self, question: str) -> GenerationResult:
        """Retrieve relevant chunks and generate an answer.

        Args:
            question: The user's natural-language question.

        Returns:
            A GenerationResult containing the answer and source metadata.
        """
        logger.info("Query: %s", question)

        # 1. Retrieve
        results = self._retriever.retrieve(
            query=question,
            top_k=self._config.retrieval.top_k,
            score_threshold=self._config.retrieval.score_threshold,
        )

        if not results:
            logger.warning("No relevant chunks found for query")

        # 2. Generate
        generation = self._llm_client.generate(
            query=question,
            search_results=results,
        )

        logger.info("Generated answer (%d chars, %d sources)",
                     len(generation.answer), len(generation.sources))
        return generation
