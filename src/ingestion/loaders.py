"""Document loaders for various file types.

Each loader reads a file and returns a Document with cleaned text and source
metadata. The load_directory function auto-detects file types and dispatches
to the appropriate loader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    """A loaded document with its text content and provenance metadata."""

    text: str
    metadata: dict = field(default_factory=dict)


def load_txt(path: Path) -> list[Document]:
    """Load a plain-text file.

    Args:
        path: Path to the .txt file.

    Returns:
        A list containing a single Document.
    """
    text = path.read_text(encoding="utf-8")
    return [
        Document(
            text=text.strip(),
            metadata={"source_file": str(path.name), "file_type": "txt"},
        )
    ]


def load_md(path: Path) -> list[Document]:
    """Load a Markdown file.

    Args:
        path: Path to the .md file.

    Returns:
        A list containing a single Document.
    """
    text = path.read_text(encoding="utf-8")
    return [
        Document(
            text=text.strip(),
            metadata={"source_file": str(path.name), "file_type": "md"},
        )
    ]


def load_pdf(path: Path) -> list[Document]:
    """Load a PDF file using PyMuPDF, producing one Document per page.

    Args:
        path: Path to the .pdf file.

    Returns:
        A list of Documents, one per page with page_number in metadata.
    """
    try:
        import pymupdf  # noqa: F811
    except ImportError as e:
        raise ImportError(
            "pymupdf is required for PDF loading. Install it with: pip install pymupdf"
        ) from e

    documents: list[Document] = []
    doc = pymupdf.open(str(path))
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source_file": str(path.name),
                            "file_type": "pdf",
                            "page_number": page_num + 1,
                        },
                    )
                )
    finally:
        doc.close()

    return documents


# Dispatch table mapping extensions to loader functions
_LOADERS: dict[str, Callable[[Path], list[Document]]] = {
    ".txt": load_txt,
    ".md": load_md,
    ".pdf": load_pdf,
}


def load_file(path: Path) -> list[Document]:
    """Load a single file, auto-detecting the format from its extension.

    Args:
        path: Path to the file.

    Returns:
        A list of Documents extracted from the file.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )
    return loader(path)


def load_directory(dir_path: Path) -> list[Document]:
    """Load all supported files from a directory (non-recursive).

    Args:
        dir_path: Path to the directory containing source documents.

    Returns:
        A flat list of Documents from all supported files.
    """
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    documents: list[Document] = []
    for file_path in sorted(dir_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.info("Loading %s", file_path.name)
            documents.extend(load_file(file_path))
        else:
            logger.debug("Skipping %s (unsupported or directory)", file_path.name)

    logger.info("Loaded %d document(s) from %s", len(documents), dir_path)
    return documents
