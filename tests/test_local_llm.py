"""Tests for the local LLM client (Ollama) and generation factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import GenerationConfig
from src.generation.factory import get_llm_client
from src.generation.llm import GenerationResult
from src.retrieval.vector_store import SearchResult


class TestLocalLLMClient:
    """Test LocalLLMClient with requests.post mocked out."""

    @patch("src.generation.local_llm.requests.post")
    def test_generate_returns_answer(self, mock_post: MagicMock) -> None:
        from src.generation.local_llm import LocalLLMClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "RAG is retrieval-augmented generation.",
            }
        }
        mock_post.return_value = mock_response

        client = LocalLLMClient(model="llama3.2")
        results = [
            SearchResult(
                chunk_text="RAG combines retrieval with generation.",
                score=0.9,
                metadata={"source_file": "doc.txt", "chunk_index": 0},
            )
        ]

        generation = client.generate("What is RAG?", results)

        assert isinstance(generation, GenerationResult)
        assert generation.answer == "RAG is retrieval-augmented generation."
        assert len(generation.sources) == 1
        assert generation.sources[0]["source_file"] == "doc.txt"

    @patch("src.generation.local_llm.requests.post")
    def test_generate_deduplicates_sources(self, mock_post: MagicMock) -> None:
        from src.generation.local_llm import LocalLLMClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Answer text"}
        }
        mock_post.return_value = mock_response

        client = LocalLLMClient()

        # Same source appearing twice
        results = [
            SearchResult(
                chunk_text="chunk A",
                score=0.9,
                metadata={"source_file": "doc.txt", "chunk_index": 0},
            ),
            SearchResult(
                chunk_text="chunk A again",
                score=0.8,
                metadata={"source_file": "doc.txt", "chunk_index": 0},
            ),
        ]

        generation = client.generate("question", results)
        assert len(generation.sources) == 1

    @patch("src.generation.local_llm.requests.post")
    def test_generate_sends_correct_payload(self, mock_post: MagicMock) -> None:
        from src.generation.local_llm import LocalLLMClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "answer"}
        }
        mock_post.return_value = mock_response

        client = LocalLLMClient(model="llama3.2", temperature=0.5, max_tokens=512)
        client.generate("test query", [])

        # Verify the call was made to the correct URL
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/chat"

        # Verify payload structure
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3.2"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 512
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    @patch("src.generation.local_llm.requests.post")
    def test_generate_handles_empty_response(self, mock_post: MagicMock) -> None:
        from src.generation.local_llm import LocalLLMClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": ""}}
        mock_post.return_value = mock_response

        client = LocalLLMClient()
        generation = client.generate("query", [])

        assert generation.answer == ""
        assert generation.sources == []

    @patch("src.generation.local_llm.requests.post")
    def test_generate_raises_on_http_error(self, mock_post: MagicMock) -> None:
        import requests

        from src.generation.local_llm import LocalLLMClient

        mock_post.return_value.raise_for_status.side_effect = (
            requests.exceptions.HTTPError("500 Server Error")
        )

        client = LocalLLMClient()
        with pytest.raises(requests.exceptions.HTTPError):
            client.generate("query", [])

    @patch("src.generation.local_llm.requests.post")
    def test_custom_base_url(self, mock_post: MagicMock) -> None:
        from src.generation.local_llm import LocalLLMClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "answer"}
        }
        mock_post.return_value = mock_response

        client = LocalLLMClient(base_url="http://myhost:9999/")
        client.generate("query", [])

        call_url = mock_post.call_args[0][0]
        assert call_url == "http://myhost:9999/api/chat"


class TestGenerationFactory:
    """Test that the factory routes to the correct LLM client."""

    @patch("src.generation.local_llm.requests")
    def test_factory_returns_local_client(self, mock_requests: MagicMock) -> None:
        from src.generation.local_llm import LocalLLMClient

        config = GenerationConfig(provider="local", model="llama3.2")
        client = get_llm_client(config)
        assert isinstance(client, LocalLLMClient)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("src.generation.llm.OpenAI")
    def test_factory_returns_openai_client(self, mock_openai_cls: MagicMock) -> None:
        from src.generation.llm import LLMClient

        config = GenerationConfig(provider="openai", model="gpt-4o-mini")
        client = get_llm_client(config)
        assert isinstance(client, LLMClient)

    def test_factory_raises_on_unknown_provider(self) -> None:
        config = GenerationConfig(provider="unknown_provider", model="test")
        with pytest.raises(ValueError, match="Unknown generation provider"):
            get_llm_client(config)
