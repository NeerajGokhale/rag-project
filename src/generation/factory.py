"""Factory function for LLM clients.

Returns the correct LLM client based on the config, making it easy to
swap providers without touching any other code.
"""

from __future__ import annotations

from src.config import GenerationConfig


def get_llm_client(config: GenerationConfig):
    """Create an LLM client from configuration.

    Args:
        config: The generation section of the application config.

    Returns:
        An initialized LLM client (LLMClient or LocalLLMClient).

    Raises:
        ValueError: If the configured provider is not recognized.
    """
    if config.provider == "openai":
        from src.generation.llm import LLMClient

        return LLMClient(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif config.provider == "local":
        from src.generation.local_llm import LocalLLMClient

        return LocalLLMClient(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    else:
        raise ValueError(
            f"Unknown generation provider '{config.provider}'. "
            f"Supported providers: openai, local"
        )
