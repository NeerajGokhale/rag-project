"""Centralized configuration loader.

Reads config.yaml for tunable parameters and .env for secrets.
All other modules import from here — no module reads env vars or config directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Project root is the parent of the directory containing this file (src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    overlap: int = 50


@dataclass
class EmbeddingsConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536


@dataclass
class RetrievalConfig:
    top_k: int = 5
    score_threshold: float = 0.5


@dataclass
class GenerationConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class VectorStoreConfig:
    provider: str = "chroma"
    persist_dir: str = "data/processed/chroma_db"
    collection_name: str = "rag_documents"


@dataclass
class Config:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)


def _build_sub_config(cls: type, data: dict[str, Any] | None) -> Any:
    """Build a dataclass instance from a dict, ignoring unknown keys."""
    if data is None:
        return cls()
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from config.yaml and .env.

    Args:
        config_path: Explicit path to config.yaml. Defaults to PROJECT_ROOT/config.yaml.

    Returns:
        A fully populated Config dataclass.
    """
    # Load .env (no-op if file doesn't exist)
    load_dotenv(PROJECT_ROOT / ".env")

    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    raw: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        logger.warning("config.yaml not found at %s — using defaults", config_path)

    return Config(
        chunking=_build_sub_config(ChunkingConfig, raw.get("chunking")),
        embeddings=_build_sub_config(EmbeddingsConfig, raw.get("embeddings")),
        retrieval=_build_sub_config(RetrievalConfig, raw.get("retrieval")),
        generation=_build_sub_config(GenerationConfig, raw.get("generation")),
        vector_store=_build_sub_config(VectorStoreConfig, raw.get("vector_store")),
    )
