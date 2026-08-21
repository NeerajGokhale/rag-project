"""Tests for document loading and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.loaders import Document, load_directory, load_file, load_md, load_txt
from src.ingestion.chunker import Chunk, chunk_document, chunk_documents


# ──────────────────────────────────────────────────────────────────────────
# Loader tests
# ──────────────────────────────────────────────────────────────────────────


class TestLoadTxt:
    def test_loads_text_content(self, sample_txt_file: Path) -> None:
        docs = load_txt(sample_txt_file)
        assert len(docs) == 1
        assert "machine learning" in docs[0].text.lower()

    def test_metadata_contains_source(self, sample_txt_file: Path) -> None:
        docs = load_txt(sample_txt_file)
        assert docs[0].metadata["source_file"] == sample_txt_file.name
        assert docs[0].metadata["file_type"] == "txt"


class TestLoadMd:
    def test_loads_markdown_content(self, sample_md_file: Path) -> None:
        docs = load_md(sample_md_file)
        assert len(docs) == 1
        assert "retrieval-augmented generation" in docs[0].text.lower()

    def test_metadata_contains_source(self, sample_md_file: Path) -> None:
        docs = load_md(sample_md_file)
        assert docs[0].metadata["source_file"] == sample_md_file.name
        assert docs[0].metadata["file_type"] == "md"


class TestLoadFile:
    def test_auto_detects_txt(self, sample_txt_file: Path) -> None:
        docs = load_file(sample_txt_file)
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "txt"

    def test_auto_detects_md(self, sample_md_file: Path) -> None:
        docs = load_file(sample_md_file)
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "md"

    def test_rejects_unsupported(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "data.csv"
        bad_file.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(bad_file)


class TestLoadDirectory:
    def test_loads_all_supported_files(self, sample_data_dir: Path) -> None:
        docs = load_directory(sample_data_dir)
        assert len(docs) == 2  # one txt + one md

    def test_skips_unsupported_files(self, sample_data_dir: Path) -> None:
        (sample_data_dir / "ignore.csv").write_text("a,b,c")
        docs = load_directory(sample_data_dir)
        assert len(docs) == 2  # csv is skipped

    def test_raises_on_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_directory(tmp_path / "nonexistent")


# ──────────────────────────────────────────────────────────────────────────
# Chunker tests
# ──────────────────────────────────────────────────────────────────────────


class TestChunkDocument:
    def test_short_doc_produces_single_chunk(self) -> None:
        doc = Document(text="Hello world", metadata={"source_file": "test.txt"})
        chunks = chunk_document(doc, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == "Hello world"

    def test_chunk_metadata_preserved(self) -> None:
        doc = Document(
            text="Some text here",
            metadata={"source_file": "doc.txt", "file_type": "txt"},
        )
        chunks = chunk_document(doc, chunk_size=100, overlap=10)
        assert chunks[0].metadata["source_file"] == "doc.txt"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_overlap_produces_more_chunks(self) -> None:
        # Create a document with enough text to produce multiple chunks
        doc = Document(
            text=" ".join(["word"] * 200),
            metadata={"source_file": "big.txt"},
        )
        chunks_no_overlap = chunk_document(doc, chunk_size=50, overlap=0)
        chunks_with_overlap = chunk_document(doc, chunk_size=50, overlap=10)
        # Overlap should produce more chunks than no overlap
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_chunk_indices_sequential(self) -> None:
        doc = Document(
            text=" ".join(["word"] * 200),
            metadata={"source_file": "test.txt"},
        )
        chunks = chunk_document(doc, chunk_size=50, overlap=10)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_empty_document(self) -> None:
        doc = Document(text="", metadata={"source_file": "empty.txt"})
        chunks = chunk_document(doc, chunk_size=100, overlap=10)
        assert len(chunks) == 0

    def test_invalid_chunk_size_raises(self) -> None:
        doc = Document(text="test", metadata={})
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_document(doc, chunk_size=0, overlap=0)

    def test_overlap_exceeds_chunk_size_raises(self) -> None:
        doc = Document(text="test", metadata={})
        with pytest.raises(ValueError, match="overlap.*must be less than"):
            chunk_document(doc, chunk_size=10, overlap=10)


class TestChunkDocuments:
    def test_chunks_multiple_documents(self) -> None:
        docs = [
            Document(text="First document content", metadata={"source_file": "a.txt"}),
            Document(text="Second document content", metadata={"source_file": "b.txt"}),
        ]
        chunks = chunk_documents(docs, chunk_size=100, overlap=10)
        assert len(chunks) == 2
        sources = {c.metadata["source_file"] for c in chunks}
        assert sources == {"a.txt", "b.txt"}
