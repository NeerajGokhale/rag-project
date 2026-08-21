"""Shared test fixtures and mock utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.config import (
    ChunkingConfig,
    Config,
    EmbeddingsConfig,
    GenerationConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from src.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for testing.

    Generates a simple hash-based vector so that similar texts produce
    similar (but not identical) embeddings.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic pseudo-embeddings from text hashes."""
        return [self._text_to_vector(t) for t in texts]

    def _text_to_vector(self, text: str) -> list[float]:
        """Convert text to a deterministic unit vector."""
        import hashlib
        import struct

        # Use SHA-256 to generate enough bytes for our vector
        h = hashlib.sha256(text.encode()).digest()
        # Expand the hash to fill the dimension
        raw_bytes = h * ((self._dimension * 4 // len(h)) + 1)

        values: list[float] = []
        for i in range(self._dimension):
            # Use 4 bytes to generate a float
            byte_chunk = raw_bytes[i * 4 : (i + 1) * 4]
            val = struct.unpack("f", byte_chunk)[0]
            # Clamp to reasonable range
            if not (-1e6 < val < 1e6):
                val = float(i) / self._dimension
            values.append(val)

        # Normalize to unit vector (cosine similarity works best with unit vectors)
        magnitude = sum(v * v for v in values) ** 0.5
        if magnitude > 0:
            values = [v / magnitude for v in values]

        return values


@pytest.fixture
def mock_embedding_provider() -> MockEmbeddingProvider:
    """Provide a mock embedding provider."""
    return MockEmbeddingProvider(dimension=64)


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Provide a test configuration pointing to a temp directory."""
    return Config(
        chunking=ChunkingConfig(chunk_size=50, overlap=10),
        embeddings=EmbeddingsConfig(provider="mock", model="mock", dimension=64),
        retrieval=RetrievalConfig(top_k=3, score_threshold=0.0),
        generation=GenerationConfig(provider="openai", model="gpt-4o-mini"),
        vector_store=VectorStoreConfig(
            provider="chroma",
            persist_dir=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        ),
    )


@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a sample .txt file for testing."""
    content = (
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from data without being explicitly programmed. "
        "Common approaches include supervised learning, unsupervised learning, "
        "and reinforcement learning."
    )
    file_path = tmp_path / "test_doc.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_md_file(tmp_path: Path) -> Path:
    """Create a sample .md file for testing."""
    content = (
        "# Introduction to RAG\n\n"
        "Retrieval-Augmented Generation combines search with language models. "
        "It retrieves relevant documents and uses them as context for generation."
    )
    file_path = tmp_path / "test_doc.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_data_dir(tmp_path: Path, sample_txt_file: Path, sample_md_file: Path) -> Path:
    """Create a directory with multiple sample files for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Copy files into the data dir
    (data_dir / "doc1.txt").write_text(sample_txt_file.read_text(), encoding="utf-8")
    (data_dir / "doc2.md").write_text(sample_md_file.read_text(), encoding="utf-8")

    return data_dir
