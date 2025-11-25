"""Memory System - A general and codebase-specific memory solution with LLM-based retrieval."""

__version__ = "0.1.0"

from .general.memory_system import MemorySystem
from .general.prefilter import EmbeddingPrefilter, EmbeddingConfig, create_openai_embedding_fn
from .codebase.code_memory_system import CodeMemorySystem

__all__ = [
    "MemorySystem",
    "CodeMemorySystem",
    "EmbeddingPrefilter",
    "EmbeddingConfig",
    "create_openai_embedding_fn"
]
