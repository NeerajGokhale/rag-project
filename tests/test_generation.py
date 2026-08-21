"""Tests for prompt building and LLM generation (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.generation.prompts import SYSTEM_PROMPT, build_context_prompt
from src.retrieval.vector_store import SearchResult


class TestBuildContextPrompt:
    def test_includes_query(self) -> None:
        prompt = build_context_prompt("What is RAG?", [])
        assert "What is RAG?" in prompt

    def test_includes_chunk_text(self) -> None:
        results = [
            SearchResult(
                chunk_text="RAG combines retrieval with generation.",
                score=0.9,
                metadata={"source_file": "doc.txt", "chunk_index": 0},
            )
        ]
        prompt = build_context_prompt("What is RAG?", results)
        assert "RAG combines retrieval with generation." in prompt

    def test_includes_source_metadata(self) -> None:
        results = [
            SearchResult(
                chunk_text="Some text",
                score=0.85,
                metadata={
                    "source_file": "report.pdf",
                    "page_number": 3,
                    "chunk_index": 2,
                },
            )
        ]
        prompt = build_context_prompt("query", results)
        assert "[Source: report.pdf, Page 3, Chunk 2]" in prompt

    def test_empty_results_shows_no_context(self) -> None:
        prompt = build_context_prompt("query", [])
        assert "No relevant context was found" in prompt

    def test_multiple_chunks_delimited(self) -> None:
        results = [
            SearchResult(
                chunk_text="First chunk",
                score=0.9,
                metadata={"source_file": "a.txt", "chunk_index": 0},
            ),
            SearchResult(
                chunk_text="Second chunk",
                score=0.8,
                metadata={"source_file": "b.txt", "chunk_index": 0},
            ),
        ]
        prompt = build_context_prompt("query", results)
        assert "First chunk" in prompt
        assert "Second chunk" in prompt
        # Each chunk should be delimited
        assert prompt.count("---") >= 4  # 2 chunks × 2 delimiters each


class TestSystemPrompt:
    def test_system_prompt_is_not_empty(self) -> None:
        assert len(SYSTEM_PROMPT) > 0

    def test_system_prompt_mentions_context(self) -> None:
        assert "context" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_sources(self) -> None:
        assert "source" in SYSTEM_PROMPT.lower()


class TestLLMClient:
    """Test LLM client with mocked OpenAI API."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("src.generation.llm.OpenAI")
    def test_generate_returns_answer(self, mock_openai_cls: MagicMock) -> None:
        from src.generation.llm import LLMClient

        # Mock the OpenAI response
        mock_choice = MagicMock()
        mock_choice.message.content = "RAG is retrieval-augmented generation."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        client = LLMClient(model="gpt-4o-mini")
        results = [
            SearchResult(
                chunk_text="RAG combines retrieval with generation.",
                score=0.9,
                metadata={"source_file": "doc.txt", "chunk_index": 0},
            )
        ]

        generation = client.generate("What is RAG?", results)

        assert generation.answer == "RAG is retrieval-augmented generation."
        assert len(generation.sources) == 1
        assert generation.sources[0]["source_file"] == "doc.txt"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("src.generation.llm.OpenAI")
    def test_generate_deduplicates_sources(self, mock_openai_cls: MagicMock) -> None:
        from src.generation.llm import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = "Answer text"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        client = LLMClient()

        # Same source appearing twice
        results = [
            SearchResult(
                chunk_text="chunk A", score=0.9,
                metadata={"source_file": "doc.txt", "chunk_index": 0},
            ),
            SearchResult(
                chunk_text="chunk A again", score=0.8,
                metadata={"source_file": "doc.txt", "chunk_index": 0},
            ),
        ]

        generation = client.generate("question", results)
        # Sources should be deduplicated
        assert len(generation.sources) == 1
