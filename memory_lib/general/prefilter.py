"""Embedding-based prefiltering for memory retrieval.

This module provides fast candidate selection using vector embeddings,
reducing the number of memories that need to be scored by the LLM.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding-based prefiltering.

    Attributes:
        enabled: Whether prefiltering is enabled.
        top_k: Number of candidates to select before LLM scoring.
        embedding_model: Name of the embedding model to use.
        embedding_dim: Dimension of the embedding vectors.
    """
    enabled: bool = True
    top_k: int = 100
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072


class EmbeddingPrefilter:
    """Fast candidate selection using embedding similarity.

    This class provides embedding-based prefiltering to reduce the number
    of memories that need to be scored by the LLM. It uses cosine similarity
    to find the most similar memories to a query.

    The typical workflow is:
    1. Store memories with embeddings
    2. When querying, embed the query
    3. Find top-K most similar memories by cosine similarity
    4. Pass only these candidates to LLM scoring

    This reduces costs from O(N) LLM calls to O(K) where K << N.

    Example:
        >>> async def embed_fn(texts):
        ...     # Call your embedding API
        ...     return [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        >>>
        >>> prefilter = EmbeddingPrefilter(embedding_fn=embed_fn)
        >>> candidates = await prefilter.get_candidates(
        ...     query="Python programming",
        ...     memories=all_memories,
        ...     top_k=100
        ... )
    """

    def __init__(
        self,
        embedding_fn: Callable[[List[str]], Awaitable[List[List[float]]]],
        config: Optional[EmbeddingConfig] = None
    ):
        """Initialize the prefilter.

        Args:
            embedding_fn: Async function that takes a list of texts and returns
                a list of embedding vectors. Each vector should be a list of floats.
            config: Optional configuration. Defaults to EmbeddingConfig().
        """
        self.embedding_fn = embedding_fn
        self.config = config or EmbeddingConfig()

    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        embeddings = await self.embedding_fn([text])
        return embeddings[0]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        return await self.embedding_fn(texts)

    def cosine_similarity(
        self,
        query_embedding: Union[List[float], np.ndarray],
        memory_embeddings: Union[List[List[float]], np.ndarray]
    ) -> np.ndarray:
        """Compute cosine similarity between query and all memories.

        Args:
            query_embedding: The query embedding vector.
            memory_embeddings: Matrix of memory embeddings (N x D).

        Returns:
            Array of similarity scores (N,).
        """
        query = np.array(query_embedding)
        memories = np.array(memory_embeddings)

        # Normalize vectors
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        memories_norm = memories / (np.linalg.norm(memories, axis=1, keepdims=True) + 1e-10)

        # Compute cosine similarity
        similarities = np.dot(memories_norm, query_norm)

        return similarities

    async def get_candidates(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get top-K candidate memories by embedding similarity.

        Args:
            query: The query string to find similar memories for.
            memories: List of memory dictionaries. Each must have an 'embedding'
                key with the pre-computed embedding vector.
            top_k: Number of candidates to return. Defaults to config.top_k.

        Returns:
            List of top-K most similar memories, sorted by similarity (highest first).
            Each memory dict will have an added 'embedding_similarity' key.
        """
        if not memories:
            return []

        top_k = top_k or self.config.top_k

        # If we have fewer memories than top_k, return all
        if len(memories) <= top_k:
            return memories

        # Filter to only memories with embeddings
        memories_with_embeddings = [
            m for m in memories
            if m.get("embedding") is not None
        ]

        if not memories_with_embeddings:
            # No embeddings available, return first top_k memories
            return memories[:top_k]

        # Embed the query
        query_embedding = await self.embed_text(query)

        # Get memory embeddings as matrix
        memory_embeddings = [m["embedding"] for m in memories_with_embeddings]

        # Compute similarities
        similarities = self.cosine_similarity(query_embedding, memory_embeddings)

        # Get top-K indices
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        # Build result with similarity scores
        candidates = []
        for idx in top_indices:
            memory = memories_with_embeddings[idx].copy()
            memory["embedding_similarity"] = float(similarities[idx])
            candidates.append(memory)

        return candidates

    async def get_candidates_from_stored(
        self,
        query: str,
        memory_ids: List[str],
        embedding_lookup: Callable[[str], Optional[List[float]]],
        memory_lookup: Callable[[str], Optional[Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get candidates using stored embeddings from a lookup function.

        This is useful when embeddings are stored separately (e.g., in a database)
        and need to be retrieved on demand.

        Args:
            query: The query string.
            memory_ids: List of memory IDs to consider.
            embedding_lookup: Function to get embedding for a memory ID.
            memory_lookup: Function to get full memory dict for an ID.
            top_k: Number of candidates to return.

        Returns:
            List of top-K candidate memories.
        """
        # Collect memories with their embeddings
        memories = []
        for mid in memory_ids:
            memory = memory_lookup(mid)
            if memory:
                embedding = embedding_lookup(mid)
                if embedding:
                    memory["embedding"] = embedding
                memories.append(memory)

        return await self.get_candidates(query, memories, top_k)


def create_openai_embedding_fn(
    api_key: str,
    model: str = "text-embedding-3-small"
) -> Callable[[List[str]], Awaitable[List[List[float]]]]:
    """Create an embedding function using OpenAI's API.

    Args:
        api_key: OpenAI API key.
        model: Embedding model name. Default: "text-embedding-3-small"

    Returns:
        Async function that embeds a list of texts.

    Example:
        >>> embed_fn = create_openai_embedding_fn(api_key="sk-...")
        >>> embeddings = await embed_fn(["Hello", "World"])
    """
    import openai

    client = openai.AsyncOpenAI(api_key=api_key)

    async def embed_texts(texts: List[str]) -> List[List[float]]:
        response = await client.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]

    return embed_texts
