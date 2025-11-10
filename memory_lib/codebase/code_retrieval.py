"""Code-specific retrieval with caching and dependency awareness.

This module extends the general retrieval pipeline with optimizations specific
to code memories, including caching, dependency boosting, and recency weighting.
"""

import asyncio
import re
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set
from dataclasses import dataclass, field
from datetime import datetime
from ..general.retrieval import MemoryRetrieval, ScoredMemory


@dataclass
class CodeContext:
    """Context information for code memory retrieval.

    Attributes:
        user_query: The user's task or question.
        current_file: File currently being edited (if any).
        recent_changes: Description of recent code changes.
        errors: Any active error messages or stack traces.
        accessed_files: List of recently accessed file paths.
        additional_context: Any other relevant context information.
    """
    user_query: str
    current_file: Optional[str] = None
    recent_changes: Optional[str] = None
    errors: Optional[str] = None
    accessed_files: List[str] = field(default_factory=list)
    additional_context: Optional[str] = None

    def to_context_string(self) -> str:
        """Convert context to a formatted string for prompting."""
        parts = [f"Task: {self.user_query}"]

        if self.current_file:
            parts.append(f"Current file: {self.current_file}")

        if self.recent_changes:
            parts.append(f"Recent changes: {self.recent_changes}")

        if self.errors:
            parts.append(f"Errors: {self.errors}")

        if self.accessed_files:
            parts.append(f"Recently accessed: {', '.join(self.accessed_files)}")

        if self.additional_context:
            parts.append(f"Additional context: {self.additional_context}")

        return "\n".join(parts)


class ScoreCache:
    """Cache for relevance scores to avoid re-scoring unchanged memories.

    Attributes:
        cache: Dictionary mapping (context_hash, memory_id, file_hash) to scores.
        max_age: Maximum age in seconds before cache entries expire.
    """

    def __init__(self, max_age: float = 3600):
        """Initialize the score cache.

        Args:
            max_age: Maximum age in seconds for cache entries. Default: 3600 (1 hour)
        """
        self.cache: Dict[tuple, tuple[float, str, float]] = {}
        self.max_age = max_age

    def get(
        self,
        context_hash: str,
        memory_id: str,
        file_hash: str
    ) -> Optional[tuple[float, str]]:
        """Retrieve a cached score.

        Args:
            context_hash: Hash of the context string.
            memory_id: ID of the memory.
            file_hash: Hash of the source file content.

        Returns:
            Tuple of (score, reasoning) if cached and not expired, else None.
        """
        key = (context_hash, memory_id, file_hash)

        if key in self.cache:
            score, reasoning, timestamp = self.cache[key]

            # Check if expired
            if time.time() - timestamp < self.max_age:
                return score, reasoning
            else:
                # Remove expired entry
                del self.cache[key]

        return None

    def set(
        self,
        context_hash: str,
        memory_id: str,
        file_hash: str,
        score: float,
        reasoning: str
    ) -> None:
        """Store a score in the cache.

        Args:
            context_hash: Hash of the context string.
            memory_id: ID of the memory.
            file_hash: Hash of the source file content.
            score: Relevance score.
            reasoning: Reasoning for the score.
        """
        key = (context_hash, memory_id, file_hash)
        self.cache[key] = (score, reasoning, time.time())

    def invalidate_memory(self, memory_id: str) -> None:
        """Invalidate all cached scores for a specific memory.

        Args:
            memory_id: ID of the memory to invalidate.
        """
        keys_to_remove = [
            key for key in self.cache.keys()
            if key[1] == memory_id
        ]

        for key in keys_to_remove:
            del self.cache[key]

    def invalidate_file(self, file_path: str) -> None:
        """Invalidate all cached scores for memories from a specific file.

        Note: This requires tracking file paths separately. For simplicity,
        this implementation clears the entire cache. In production, maintain
        a file_path -> memory_id mapping.

        Args:
            file_path: Path to the file whose memories should be invalidated.
        """
        # In a production system, maintain a separate index of file_path -> memory_ids
        # For now, just clear the cache
        self.cache.clear()

    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()

    def size(self) -> int:
        """Get the number of cached entries."""
        return len(self.cache)


class CodeMemoryRetrieval(MemoryRetrieval):
    """Extended retrieval for code memories with optimizations.

    This class extends the general MemoryRetrieval with code-specific features:
    - Caching of relevance scores for unchanged code
    - Dependency awareness (boosting related entities)
    - Recency weighting (recently modified files get slight boost)
    - File-level grouping (combine entities from same file)

    Example:
        >>> async def small_model(prompt):
        ...     return "Score: 0.8\\nReason: Relevant"
        >>> retrieval = CodeMemoryRetrieval(
        ...     small_model_fn=small_model,
        ...     enable_caching=True,
        ...     enable_dependency_boost=True
        ... )
        >>> context = CodeContext(
        ...     user_query="Fix the authentication bug",
        ...     current_file="api/auth.py",
        ...     errors="AttributeError in validate_token"
        ... )
        >>> memories = await retrieval.retrieve_code_memories(context, all_memories)
    """

    def __init__(
        self,
        small_model_fn: Callable[[str], Awaitable[str]],
        relevance_threshold: float = 0.7,
        max_memories: int = 15,
        batch_size: int = 10,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        enable_caching: bool = True,
        cache_max_age: float = 3600,
        enable_dependency_boost: bool = True,
        dependency_boost_factor: float = 0.15,
        enable_recency_boost: bool = True,
        recency_boost_factor: float = 0.1
    ):
        """Initialize code memory retrieval.

        Args:
            small_model_fn: Async function for calling small LLM.
            relevance_threshold: Minimum relevance score (0-1). Default: 0.7
            max_memories: Maximum memories to return (K). Default: 15 (higher for code)
            batch_size: Parallel API calls. Default: 10
            retry_attempts: Number of retries. Default: 3
            retry_delay: Initial retry delay. Default: 1.0
            enable_caching: Whether to cache scores. Default: True
            cache_max_age: Cache entry max age in seconds. Default: 3600
            enable_dependency_boost: Boost scores for dependencies. Default: True
            dependency_boost_factor: Amount to boost dependency scores. Default: 0.15
            enable_recency_boost: Boost scores for recently modified files. Default: True
            recency_boost_factor: Amount to boost recent files. Default: 0.1
        """
        super().__init__(
            small_model_fn=small_model_fn,
            relevance_threshold=relevance_threshold,
            max_memories=max_memories,
            batch_size=batch_size,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay
        )

        self.enable_caching = enable_caching
        self.cache = ScoreCache(max_age=cache_max_age) if enable_caching else None

        self.enable_dependency_boost = enable_dependency_boost
        self.dependency_boost_factor = dependency_boost_factor

        self.enable_recency_boost = enable_recency_boost
        self.recency_boost_factor = recency_boost_factor

    def _create_code_scoring_prompt(
        self,
        code_context: CodeContext,
        memory: Dict[str, Any]
    ) -> str:
        """Create a specialized prompt for scoring code memories.

        Args:
            code_context: Current coding context.
            memory: Code memory dictionary.

        Returns:
            Formatted prompt for the small model.
        """
        context_str = code_context.to_context_string()

        # Build memory description
        if memory.get("type") == "code":
            memory_str = f"""File: {memory.get('file_path', 'Unknown')}
Entity: {memory.get('entity_name', 'Unknown')}
Signature: {memory.get('signature', 'N/A')}
Code snippet: {memory.get('code_snippet', '')[:200]}...
Docstring: {memory.get('docstring', 'N/A')}
Dependencies: {', '.join(memory.get('dependencies', []))}
Language: {memory.get('language', 'Unknown')}"""
        else:
            # Non-code memory
            memory_str = f"""Category: {memory.get('category', 'Unknown')}
Title: {memory.get('title', 'N/A')}
Content: {memory.get('content', '')[:200]}..."""

        prompt = f"""{context_str}

Code memory:
{memory_str}

Rate relevance 0-1. Consider:
- Direct dependencies and imports
- Similar patterns or logic
- Related functionality
- Error handling for current issue
- File proximity to current work

Response format:
Score: <0 to 1>
Reason: <brief explanation>"""

        return prompt

    async def _score_code_memory_with_cache(
        self,
        code_context: CodeContext,
        memory: Dict[str, Any],
        file_hashes: Dict[str, str]
    ) -> ScoredMemory:
        """Score a code memory with caching support.

        Args:
            code_context: Current coding context.
            memory: Code memory to score.
            file_hashes: Dictionary mapping file paths to their content hashes.

        Returns:
            ScoredMemory with relevance score.
        """
        memory_id = memory["id"]
        file_path = memory.get("file_path", "")
        file_hash = file_hashes.get(file_path, "")

        # Check cache
        if self.enable_caching and self.cache:
            context_hash = str(hash(code_context.to_context_string()))
            cached = self.cache.get(context_hash, memory_id, file_hash)

            if cached:
                score, reasoning = cached
                return ScoredMemory(
                    memory_id=memory_id,
                    text=self._format_memory_text(memory),
                    metadata=memory.get("metadata", {}),
                    timestamp=memory.get("timestamp", ""),
                    relevance_score=score,
                    reasoning=f"[CACHED] {reasoning}"
                )

        # Score using small model
        prompt = self._create_code_scoring_prompt(code_context, memory)

        for attempt in range(self.retry_attempts):
            try:
                response = await self.small_model_fn(prompt)
                score, reasoning = self._parse_score_response(response)

                # Cache the result
                if self.enable_caching and self.cache:
                    context_hash = str(hash(code_context.to_context_string()))
                    self.cache.set(context_hash, memory_id, file_hash, score, reasoning)

                return ScoredMemory(
                    memory_id=memory_id,
                    text=self._format_memory_text(memory),
                    metadata=memory.get("metadata", {}),
                    timestamp=memory.get("timestamp", ""),
                    relevance_score=score,
                    reasoning=reasoning
                )

            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    return ScoredMemory(
                        memory_id=memory_id,
                        text=self._format_memory_text(memory),
                        metadata=memory.get("metadata", {}),
                        timestamp=memory.get("timestamp", ""),
                        relevance_score=0.0,
                        reasoning=f"API error: {str(e)}"
                    )

    def _format_memory_text(self, memory: Dict[str, Any]) -> str:
        """Format memory for display."""
        if memory.get("type") == "code":
            return f"{memory.get('entity_name', 'Unknown')} in {memory.get('file_path', 'Unknown')}"
        else:
            return memory.get("title", memory.get("content", "")[:100])

    def _apply_dependency_boost(
        self,
        scored_memories: List[ScoredMemory],
        memories: List[Dict[str, Any]]
    ) -> List[ScoredMemory]:
        """Apply dependency boosting to scored memories.

        If function A is relevant and calls function B, boost B's score.

        Args:
            scored_memories: List of scored memories.
            memories: Original memory dictionaries.

        Returns:
            List with adjusted scores.
        """
        if not self.enable_dependency_boost:
            return scored_memories

        # Build ID -> memory mapping
        id_to_memory = {mem["id"]: mem for mem in memories}
        id_to_scored = {sm.memory_id: sm for sm in scored_memories}

        # Find highly relevant memories (score >= threshold)
        relevant_ids = {
            sm.memory_id for sm in scored_memories
            if sm.relevance_score >= self.relevance_threshold
        }

        # Collect all dependencies from relevant memories
        boosted_entities = set()
        for mem_id in relevant_ids:
            memory = id_to_memory.get(mem_id)
            if memory and memory.get("type") == "code":
                dependencies = memory.get("dependencies", [])
                boosted_entities.update(dependencies)

        # Apply boost to memories whose entity names match dependencies
        adjusted = []
        for scored in scored_memories:
            memory = id_to_memory.get(scored.memory_id)

            if memory and memory.get("type") == "code":
                entity_name = memory.get("entity_name", "")

                if entity_name in boosted_entities:
                    # Apply boost but don't exceed 1.0
                    new_score = min(1.0, scored.relevance_score + self.dependency_boost_factor)

                    adjusted.append(ScoredMemory(
                        memory_id=scored.memory_id,
                        text=scored.text,
                        metadata=scored.metadata,
                        timestamp=scored.timestamp,
                        relevance_score=new_score,
                        reasoning=f"{scored.reasoning} [DEPENDENCY BOOST: +{self.dependency_boost_factor}]"
                    ))
                    continue

            adjusted.append(scored)

        return adjusted

    def _apply_recency_boost(
        self,
        scored_memories: List[ScoredMemory],
        memories: List[Dict[str, Any]]
    ) -> List[ScoredMemory]:
        """Apply recency boost to recently modified files.

        Args:
            scored_memories: List of scored memories.
            memories: Original memory dictionaries.

        Returns:
            List with adjusted scores.
        """
        if not self.enable_recency_boost:
            return scored_memories

        # Calculate recency threshold (e.g., last 7 days)
        from datetime import timedelta
        now = datetime.utcnow()
        recent_threshold = (now - timedelta(days=7)).isoformat()

        id_to_memory = {mem["id"]: mem for mem in memories}

        adjusted = []
        for scored in scored_memories:
            memory = id_to_memory.get(scored.memory_id)

            if memory and memory.get("type") == "code":
                last_modified = memory.get("last_modified", "")

                if last_modified and last_modified > recent_threshold:
                    # Apply recency boost
                    new_score = min(1.0, scored.relevance_score + self.recency_boost_factor)

                    adjusted.append(ScoredMemory(
                        memory_id=scored.memory_id,
                        text=scored.text,
                        metadata=scored.metadata,
                        timestamp=scored.timestamp,
                        relevance_score=new_score,
                        reasoning=f"{scored.reasoning} [RECENCY BOOST: +{self.recency_boost_factor}]"
                    ))
                    continue

            adjusted.append(scored)

        return adjusted

    async def retrieve_code_memories(
        self,
        code_context: CodeContext,
        memories: List[Dict[str, Any]],
        file_hashes: Optional[Dict[str, str]] = None
    ) -> List[ScoredMemory]:
        """Retrieve relevant code memories with all optimizations.

        This is the main entry point for code memory retrieval.

        Args:
            code_context: Current coding context.
            memories: List of all available code memories.
            file_hashes: Optional dictionary of file path -> content hash for caching.

        Returns:
            List of selected ScoredMemory objects with optimizations applied.

        Example:
            >>> context = CodeContext(
            ...     user_query="Implement user authentication",
            ...     current_file="api/auth.py"
            ... )
            >>> memories = await retrieval.retrieve_code_memories(
            ...     context,
            ...     all_memories,
            ...     file_hashes={"api/auth.py": "abc123..."}
            ... )
        """
        file_hashes = file_hashes or {}

        # Score all memories (with caching)
        scored_memories = []
        for i in range(0, len(memories), self.batch_size):
            batch = memories[i:i + self.batch_size]
            tasks = [
                self._score_code_memory_with_cache(code_context, mem, file_hashes)
                for mem in batch
            ]
            batch_scored = await asyncio.gather(*tasks)
            scored_memories.extend(batch_scored)

        # Apply dependency boost
        scored_memories = self._apply_dependency_boost(scored_memories, memories)

        # Apply recency boost
        scored_memories = self._apply_recency_boost(scored_memories, memories)

        # Filter and select top-K
        selected = self.filter_and_select(scored_memories)

        return selected

    def format_code_memories_for_prompt(
        self,
        memories: List[ScoredMemory],
        group_by_file: bool = True,
        include_scores: bool = True
    ) -> str:
        """Format code memories for inclusion in a prompt.

        Args:
            memories: List of ScoredMemory objects.
            group_by_file: Whether to group entities from the same file. Default: True
            include_scores: Whether to include relevance scores. Default: True

        Returns:
            Formatted string ready for prompt.

        Example:
            >>> formatted = retrieval.format_code_memories_for_prompt(
            ...     memories,
            ...     group_by_file=True
            ... )
        """
        if not memories:
            return "No relevant code memories found."

        if not group_by_file:
            return self.format_memories_for_prompt(memories, include_scores)

        # Group by file (extract from text)
        # This is a simplified version - in production, keep file_path in ScoredMemory
        formatted_parts = []
        for mem in memories:
            if include_scores:
                header = f'<code id="{mem.memory_id}" score="{mem.relevance_score:.2f}">'
            else:
                header = f'<code id="{mem.memory_id}">'

            formatted_parts.append(f"{header}\n{mem.text}\n</code>")

        return "\n\n".join(formatted_parts)

    def invalidate_cache_for_file(self, file_path: str) -> None:
        """Invalidate cached scores for a specific file.

        Call this when a file is modified.

        Args:
            file_path: Path to the modified file.
        """
        if self.cache:
            self.cache.invalidate_file(file_path)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache size and status.
        """
        return {
            "caching_enabled": self.enable_caching,
            "cache_size": self.cache.size() if self.cache else 0,
            "dependency_boost_enabled": self.enable_dependency_boost,
            "recency_boost_enabled": self.enable_recency_boost
        }
