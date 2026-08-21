"""Token-aware document chunker.

Splits documents into overlapping chunks using tiktoken for accurate token
counting. Preserves source provenance metadata on every chunk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import tiktoken

from src.ingestion.loaders import Document

logger = logging.getLogger(__name__)

# Default encoding used by OpenAI models
_DEFAULT_ENCODING = "cl100k_base"


@dataclass
class Chunk:
    """A chunk of text with provenance metadata."""

    text: str
    metadata: dict = field(default_factory=dict)


def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding_name: str = _DEFAULT_ENCODING,
) -> list[Chunk]:
    """Split a Document into token-sized, overlapping chunks.

    Args:
        doc: The source Document to chunk.
        chunk_size: Maximum number of tokens per chunk.
        overlap: Number of overlapping tokens between consecutive chunks.
        encoding_name: The tiktoken encoding to use for tokenization.

    Returns:
        A list of Chunk objects with inherited + chunk-specific metadata.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(doc.text)

    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        chunk_metadata = {
            **doc.metadata,
            "chunk_index": chunk_index,
        }

        chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
        chunk_index += 1

        # If we've consumed all tokens, stop
        if end == len(tokens):
            break

        start += step

    logger.debug(
        "Chunked '%s' into %d chunk(s) (chunk_size=%d, overlap=%d)",
        doc.metadata.get("source_file", "<unknown>"),
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 512,
    overlap: int = 50,
    encoding_name: str = _DEFAULT_ENCODING,
) -> list[Chunk]:
    """Chunk a list of Documents.

    Args:
        docs: The Documents to chunk.
        chunk_size: Maximum number of tokens per chunk.
        overlap: Number of overlapping tokens between consecutive chunks.
        encoding_name: The tiktoken encoding to use.

    Returns:
        A flat list of all Chunks from all Documents.
    """
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(
            chunk_document(doc, chunk_size=chunk_size, overlap=overlap,
                           encoding_name=encoding_name)
        )
    logger.info("Produced %d total chunk(s) from %d document(s)", len(all_chunks), len(docs))
    return all_chunks
