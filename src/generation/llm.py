"""LLM client for the generation step."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from openai import OpenAI

from src.retrieval.vector_store import SearchResult
from src.generation.prompts import SYSTEM_PROMPT, build_context_prompt

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """The generated answer along with the sources that were used."""

    answer: str
    sources: list[dict] = field(default_factory=list)


class LLMClient:
    """Wraps OpenAI chat completions for RAG generation.

    Assembles the prompt from retrieved context and the user query, calls the
    LLM, and returns the answer with source metadata.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Copy .env.example to .env and fill in your key."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        logger.info("Initialized LLM client (model=%s)", model)

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

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        answer = response.choices[0].message.content or ""

        # Collect unique source metadata for attribution
        sources: list[dict] = []
        seen: set[str] = set()
        for result in search_results:
            key = f"{result.metadata.get('source_file', '')}::{result.metadata.get('chunk_index', '')}"
            if key not in seen:
                seen.add(key)
                sources.append(result.metadata)

        return GenerationResult(answer=answer.strip(), sources=sources)
