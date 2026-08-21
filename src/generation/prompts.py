"""Prompt templates for the RAG generation step.

All prompt templates live here as named constants — never as inline strings
in other modules. This keeps prompt engineering centralized and auditable.
"""

from __future__ import annotations

from src.retrieval.vector_store import SearchResult

# ---------------------------------------------------------------------------
# System prompt — tells the LLM how to behave
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based ONLY on the "
    "provided context. Follow these rules:\n"
    "1. Answer the question using ONLY the information in the context below.\n"
    "2. If the context does not contain enough information to answer, say "
    '"I don\'t have enough information to answer that question."\n'
    "3. Cite your sources by referencing the [Source] tags from the context.\n"
    "4. Be concise and direct."
)

# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------
_CHUNK_TEMPLATE = (
    "---\n"
    "[Source: {source_file}"
    "{page_info}"
    ", Chunk {chunk_index}]\n"
    "{text}\n"
    "---"
)


def _format_chunk(result: SearchResult) -> str:
    """Format a single SearchResult into a delimited context block."""
    meta = result.metadata
    page_info = ""
    if "page_number" in meta:
        page_info = f", Page {meta['page_number']}"
    return _CHUNK_TEMPLATE.format(
        source_file=meta.get("source_file", "unknown"),
        page_info=page_info,
        chunk_index=meta.get("chunk_index", "?"),
        text=result.chunk_text,
    )


def build_context_prompt(query: str, search_results: list[SearchResult]) -> str:
    """Build the user-message prompt with retrieved context and the query.

    Args:
        query: The user's question.
        search_results: Retrieved chunks with scores and metadata.

    Returns:
        A formatted prompt string ready to be sent as the user message.
    """
    if not search_results:
        context_block = "(No relevant context was found.)"
    else:
        context_block = "\n\n".join(_format_chunk(r) for r in search_results)

    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above. Cite your sources."
    )
