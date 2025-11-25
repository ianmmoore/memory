"""Retrieval pipeline for memory system using LLM-based relevance scoring.

This module implements an exhaustive reasoning approach where a small language model
scores the relevance of each memory to the current context, enabling precise
memory selection.

Optionally supports embedding-based prefiltering to reduce LLM API calls.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Callable, Awaitable, TYPE_CHECKING
from dataclasses import dataclass
import time

if TYPE_CHECKING:
    from .prefilter import EmbeddingPrefilter


@dataclass
class ScoredMemory:
    """A memory with its relevance score and reasoning.

    Attributes:
        memory_id: Unique identifier of the memory.
        text: Content of the memory.
        metadata: Associated metadata dictionary.
        timestamp: ISO format timestamp of creation.
        relevance_score: Float between 0 and 1 indicating relevance.
        reasoning: Explanation of why this score was assigned.
    """
    memory_id: str
    text: str
    metadata: Dict[str, Any]
    timestamp: str
    relevance_score: float
    reasoning: str


class MemoryRetrieval:
    """Retrieval pipeline using LLM-based exhaustive reasoning.

    This class implements a sophisticated retrieval strategy:
    1. Uses a small model to score relevance of ALL memories
    2. Filters memories based on relevance threshold
    3. Selects top-K memories for inclusion in context

    The approach ensures high-quality memory selection while managing token limits
    through the filtering and top-K selection process.

    Attributes:
        small_model_fn: Async function to call the small model for scoring.
        relevance_threshold: Minimum score for a memory to be considered (0-1).
        max_memories: Maximum number of memories to return (K).
        batch_size: Number of parallel API calls to make.
        retry_attempts: Number of retries for failed API calls.
        retry_delay: Initial delay in seconds for exponential backoff.

    Example:
        >>> async def score_fn(prompt):
        ...     # Your LLM API call here
        ...     return "Score: 0.8\\nReason: Highly relevant"
        >>> retrieval = MemoryRetrieval(small_model_fn=score_fn)
        >>> memories = [{"id": "1", "text": "Python fact", "metadata": {}, "timestamp": "2024-01-01"}]
        >>> results = await retrieval.retrieve_relevant_memories(
        ...     memories=memories,
        ...     context="Tell me about Python"
        ... )
    """

    def __init__(
        self,
        small_model_fn: Callable[[str], Awaitable[str]],
        relevance_threshold: float = 0.7,
        max_memories: int = 10,
        batch_size: int = 10,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        prefilter: Optional["EmbeddingPrefilter"] = None,
        prefilter_top_k: int = 100
    ):
        """Initialize the retrieval pipeline.

        Args:
            small_model_fn: Async function that takes a prompt string and returns
                a response string. This function should call your small LLM API
                (e.g., GPT-3.5, Claude Haiku) to score relevance.
            relevance_threshold: Minimum relevance score (0-1) for a memory to be
                included. Memories below this threshold are filtered out.
                Default: 0.7
            max_memories: Maximum number of memories to return (K). If more than K
                memories pass the threshold, only the top K by score are returned.
                Default: 10
            batch_size: Number of API calls to make in parallel. Higher values
                increase speed but may hit rate limits. Default: 10
            retry_attempts: Number of times to retry failed API calls.
                Default: 3
            retry_delay: Initial delay in seconds for exponential backoff retries.
                Default: 1.0
            prefilter: Optional EmbeddingPrefilter for reducing candidate count
                before LLM scoring. If provided, only top prefilter_top_k candidates
                by embedding similarity will be scored.
            prefilter_top_k: Number of candidates to select via prefiltering before
                LLM scoring. Only used if prefilter is provided. Default: 100

        Raises:
            ValueError: If threshold is not between 0 and 1, or max_memories < 1.
        """
        if not 0 <= relevance_threshold <= 1:
            raise ValueError("relevance_threshold must be between 0 and 1")
        if max_memories < 1:
            raise ValueError("max_memories must be at least 1")

        self.small_model_fn = small_model_fn
        self.relevance_threshold = relevance_threshold
        self.max_memories = max_memories
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.prefilter = prefilter
        self.prefilter_top_k = prefilter_top_k

    def _create_scoring_prompt(self, context: str, memory: Dict[str, Any]) -> str:
        """Create a prompt for the small model to score relevance.

        Args:
            context: Current context/task description.
            memory: Memory dictionary with text and metadata.

        Returns:
            Formatted prompt string for relevance scoring.
        """
        metadata_str = ", ".join(f"{k}: {v}" for k, v in memory["metadata"].items())

        prompt = f"""Context: {context}

Memory: {memory["text"]}
Metadata: {metadata_str}
Timestamp: {memory["timestamp"]}

Is this memory relevant to the current context? Rate relevance from 0 to 1 (0 = not relevant, 1 = highly relevant).
Provide your response in this exact format:
Score: <number between 0 and 1>
Reason: <brief explanation>"""

        return prompt

    def _parse_score_response(self, response: str) -> tuple[float, str]:
        """Parse the small model's response to extract score and reasoning.

        Args:
            response: Raw response text from the small model.

        Returns:
            Tuple of (score, reasoning). Score is clamped to [0, 1].

        Note:
            If parsing fails, returns (0.0, "Failed to parse response").
        """
        try:
            # Extract score using regex
            score_match = re.search(r"Score:\s*([\d.]+)", response, re.IGNORECASE)
            if not score_match:
                return 0.0, "Failed to parse score"

            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            # Extract reason
            reason_match = re.search(r"Reason:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "No reason provided"

            return score, reason
        except Exception as e:
            return 0.0, f"Parse error: {str(e)}"

    async def _score_memory_with_retry(
        self,
        context: str,
        memory: Dict[str, Any]
    ) -> ScoredMemory:
        """Score a single memory with retry logic.

        Args:
            context: Current context/task description.
            memory: Memory dictionary to score.

        Returns:
            ScoredMemory object with relevance score and reasoning.

        Note:
            Uses exponential backoff for retries. If all retries fail,
            returns a score of 0.0.
        """
        prompt = self._create_scoring_prompt(context, memory)

        for attempt in range(self.retry_attempts):
            try:
                response = await self.small_model_fn(prompt)
                score, reasoning = self._parse_score_response(response)

                return ScoredMemory(
                    memory_id=memory["id"],
                    text=memory["text"],
                    metadata=memory["metadata"],
                    timestamp=memory["timestamp"],
                    relevance_score=score,
                    reasoning=reasoning
                )
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    return ScoredMemory(
                        memory_id=memory["id"],
                        text=memory["text"],
                        metadata=memory["metadata"],
                        timestamp=memory["timestamp"],
                        relevance_score=0.0,
                        reasoning=f"API error: {str(e)}"
                    )

    async def _score_memories_batch(
        self,
        context: str,
        memories: List[Dict[str, Any]]
    ) -> List[ScoredMemory]:
        """Score a batch of memories in parallel.

        Args:
            context: Current context/task description.
            memories: List of memory dictionaries to score.

        Returns:
            List of ScoredMemory objects.
        """
        tasks = [
            self._score_memory_with_retry(context, memory)
            for memory in memories
        ]
        return await asyncio.gather(*tasks)

    async def score_all_memories(
        self,
        context: str,
        memories: List[Dict[str, Any]]
    ) -> List[ScoredMemory]:
        """Score all memories using the small model with batching.

        This is the exhaustive reasoning step - every memory is scored
        for relevance to the current context using the small model.

        Args:
            context: Current context/task description that memories are scored against.
            memories: List of all available memories to score.

        Returns:
            List of ScoredMemory objects with relevance scores and reasoning.

        Example:
            >>> memories = [
            ...     {"id": "1", "text": "Python is interpreted", "metadata": {}, "timestamp": "2024-01-01"},
            ...     {"id": "2", "text": "Java is compiled", "metadata": {}, "timestamp": "2024-01-02"}
            ... ]
            >>> scored = await retrieval.score_all_memories("Python programming", memories)
            >>> scored[0].relevance_score > scored[1].relevance_score
            True
        """
        if not memories:
            return []

        all_scored = []

        # Process in batches to avoid overwhelming the API
        for i in range(0, len(memories), self.batch_size):
            batch = memories[i:i + self.batch_size]
            scored_batch = await self._score_memories_batch(context, batch)
            all_scored.extend(scored_batch)

        return all_scored

    def filter_and_select(
        self,
        scored_memories: List[ScoredMemory]
    ) -> List[ScoredMemory]:
        """Filter and select top-K memories based on relevance scores.

        This implements the filtering and selection logic:
        1. Filter: Keep only memories with score >= threshold
        2. If filtered set has <= K memories: return all
        3. If filtered set has > K memories: return top K by score

        Args:
            scored_memories: List of memories with relevance scores.

        Returns:
            Filtered and selected list of memories, bounded by max_memories.
            Memories are sorted by relevance score (highest first).

        Example:
            >>> scored = [
            ...     ScoredMemory("1", "text1", {}, "2024-01-01", 0.9, "Very relevant"),
            ...     ScoredMemory("2", "text2", {}, "2024-01-01", 0.5, "Somewhat relevant"),
            ...     ScoredMemory("3", "text3", {}, "2024-01-01", 0.8, "Relevant")
            ... ]
            >>> retrieval = MemoryRetrieval(lambda x: None, relevance_threshold=0.7)
            >>> selected = retrieval.filter_and_select(scored)
            >>> len(selected)
            2
            >>> selected[0].relevance_score
            0.9
        """
        # Filter by threshold
        filtered = [
            mem for mem in scored_memories
            if mem.relevance_score >= self.relevance_threshold
        ]

        # Sort by score (descending)
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)

        # Take top K
        return filtered[:self.max_memories]

    async def retrieve_relevant_memories(
        self,
        context: str,
        memories: List[Dict[str, Any]],
        use_prefilter: Optional[bool] = None
    ) -> List[ScoredMemory]:
        """Complete retrieval pipeline: score, filter, and select memories.

        This is the main entry point for memory retrieval. It runs the complete
        pipeline:
        1. (Optional) Prefilter to top-K candidates by embedding similarity
        2. Score memories using the small model
        3. Filter by relevance threshold
        4. Select top-K memories

        When prefiltering is enabled (and prefilter is configured), only the
        top prefilter_top_k candidates by embedding similarity are scored,
        dramatically reducing API costs.

        Args:
            context: Current context/task description to score memories against.
            memories: List of all available memory dictionaries.
            use_prefilter: Whether to use prefiltering. If None, uses prefilter
                if one is configured. Set to False to force exhaustive scoring.

        Returns:
            List of selected ScoredMemory objects, sorted by relevance (highest first).
            The list is bounded by max_memories and filtered by relevance_threshold.

        Example:
            >>> async def small_model(prompt):
            ...     # Simulate API call
            ...     if "Python" in prompt:
            ...         return "Score: 0.9\\nReason: Highly relevant"
            ...     return "Score: 0.3\\nReason: Not very relevant"
            >>> retrieval = MemoryRetrieval(small_model_fn=small_model)
            >>> memories = [
            ...     {"id": "1", "text": "Python is great", "metadata": {}, "timestamp": "2024-01-01"}
            ... ]
            >>> results = await retrieval.retrieve_relevant_memories(
            ...     context="Tell me about Python",
            ...     memories=memories
            ... )
            >>> len(results) > 0
            True
        """
        # Determine whether to use prefiltering
        should_prefilter = use_prefilter if use_prefilter is not None else (self.prefilter is not None)

        # Step 1: Optional prefiltering
        if should_prefilter and self.prefilter is not None:
            candidates = await self.prefilter.get_candidates(
                query=context,
                memories=memories,
                top_k=self.prefilter_top_k
            )
        else:
            candidates = memories

        # Step 2: Score candidates (exhaustive over candidates, not all memories)
        scored_memories = await self.score_all_memories(context, candidates)

        # Step 3 & 4: Filter and select top-K
        selected_memories = self.filter_and_select(scored_memories)

        return selected_memories

    def set_prefilter(
        self,
        prefilter: Optional["EmbeddingPrefilter"],
        top_k: Optional[int] = None
    ) -> None:
        """Set or update the prefilter configuration.

        Args:
            prefilter: The EmbeddingPrefilter to use, or None to disable.
            top_k: Number of candidates to prefilter to. If None, keeps current value.
        """
        self.prefilter = prefilter
        if top_k is not None:
            self.prefilter_top_k = top_k

    def format_memories_for_prompt(
        self,
        memories: List[ScoredMemory],
        include_scores: bool = True
    ) -> str:
        """Format selected memories for inclusion in a prompt.

        Args:
            memories: List of selected ScoredMemory objects.
            include_scores: Whether to include relevance scores in the output.
                Default: True

        Returns:
            Formatted string ready to include in a prompt to the primary model.

        Example:
            >>> memory = ScoredMemory(
            ...     "1", "Python uses duck typing", {"topic": "python"},
            ...     "2024-01-01", 0.95, "Very relevant"
            ... )
            >>> retrieval = MemoryRetrieval(lambda x: None)
            >>> formatted = retrieval.format_memories_for_prompt([memory])
            >>> "duck typing" in formatted
            True
        """
        if not memories:
            return "No relevant memories found."

        formatted_parts = []
        for mem in memories:
            if include_scores:
                header = f'<memory id="{mem.memory_id}" score="{mem.relevance_score:.2f}">'
            else:
                header = f'<memory id="{mem.memory_id}">'

            formatted_parts.append(f"{header}\n{mem.text}\n</memory>")

        return "\n\n".join(formatted_parts)
