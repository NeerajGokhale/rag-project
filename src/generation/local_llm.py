"""Local LLM client using Ollama's REST API.

Calls ``POST /api/chat`` on a locally-running Ollama instance.
No API key or paid service required — just install Ollama and pull a model.
"""

from __future__ import annotations

import json
import logging

import requests

from src.generation.llm import GenerationResult
from src.generation.prompts import SYSTEM_PROMPT, build_context_prompt
from src.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"


class LocalLLMClient:
    """Wraps Ollama's ``/api/chat`` endpoint for RAG generation.

    Drop-in replacement for :class:`LLMClient` — same ``generate()``
    signature and return type.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        base_url: str = _DEFAULT_OLLAMA_URL,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._base_url = base_url.rstrip("/")
        logger.info(
            "Initialized local LLM client (model=%s, url=%s)",
            model,
            self._base_url,
        )

    def generate(
        self,
        query: str,
        search_results: list[SearchResult],
    ) -> GenerationResult:
        """Generate an answer from the query and retrieved context.

        Args:
            query: The user's natural-language question.
            search_results: Retrieved chunks from the vector store.

        Returns:
            A GenerationResult with the answer text and source metadata.
        """
        user_prompt = build_context_prompt(query, search_results)
        logger.debug("Full assembled prompt:\n%s", user_prompt)

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

        url = f"{self._base_url}/api/chat"
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        answer = data.get("message", {}).get("content", "")

        # Collect unique source metadata for attribution
        sources: list[dict] = []
        seen: set[str] = set()
        for result in search_results:
            key = f"{result.metadata.get('source_file', '')}::{result.metadata.get('chunk_index', '')}"
            if key not in seen:
                seen.add(key)
                sources.append(result.metadata)

        return GenerationResult(answer=answer.strip(), sources=sources)
