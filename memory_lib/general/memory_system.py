"""Main memory system API combining storage and retrieval.

This module provides the high-level MemorySystem class that integrates storage
and LLM-based retrieval into a unified, easy-to-use API.
"""

from typing import Dict, List, Optional, Any, Callable, Awaitable
from .storage import MemoryStorage
from .retrieval import MemoryRetrieval, ScoredMemory


class MemorySystem:
    """Complete memory system with storage and LLM-based retrieval.

    This is the main entry point for using the memory system. It combines
    persistent storage with intelligent LLM-based retrieval to provide a
    complete solution for memory management.

    The system works by:
    1. Storing memories with metadata in a persistent database
    2. Using a small LLM to score relevance of all memories to a query
    3. Filtering and selecting the most relevant memories
    4. Formatting memories for use with a primary LLM

    Attributes:
        storage: The MemoryStorage instance managing persistence.
        retrieval: The MemoryRetrieval instance managing relevance scoring.

    Example:
        >>> async def small_model_api(prompt: str) -> str:
        ...     # Call your small LLM (e.g., GPT-3.5, Claude Haiku)
        ...     response = await your_llm_client.complete(prompt)
        ...     return response
        >>>
        >>> async def primary_model_api(prompt: str) -> str:
        ...     # Call your primary LLM (e.g., GPT-4, Claude Opus)
        ...     response = await your_llm_client.complete(prompt)
        ...     return response
        >>>
        >>> # Initialize the system
        >>> memory_system = MemorySystem(
        ...     small_model_fn=small_model_api,
        ...     db_path="my_memories.db",
        ...     relevance_threshold=0.7,
        ...     max_memories=10
        ... )
        >>>
        >>> # Add some memories
        >>> memory_system.add_memory(
        ...     "Python uses dynamic typing and is interpreted",
        ...     metadata={"topic": "python", "category": "language-features"}
        ... )
        >>>
        >>> # Query with context
        >>> response = await memory_system.query(
        ...     context="Explain Python's type system",
        ...     task="Provide a detailed explanation",
        ...     primary_model_fn=primary_model_api
        ... )
    """

    def __init__(
        self,
        small_model_fn: Callable[[str], Awaitable[str]],
        db_path: str = "memories.db",
        relevance_threshold: float = 0.7,
        max_memories: int = 10,
        batch_size: int = 10,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize the memory system.

        Args:
            small_model_fn: Async function for calling a small LLM to score relevance.
                Should take a prompt string and return a response string.
            db_path: Path to the SQLite database file for persistent storage.
                Default: "memories.db"
            relevance_threshold: Minimum relevance score (0-1) for memory inclusion.
                Default: 0.7
            max_memories: Maximum number of memories to return in queries (K).
                Default: 10
            batch_size: Number of parallel API calls for scoring.
                Default: 10
            retry_attempts: Number of retries for failed API calls.
                Default: 3
            retry_delay: Initial delay in seconds for exponential backoff.
                Default: 1.0

        Example:
            >>> async def my_small_model(prompt):
            ...     return await llm_api.complete(prompt, model="fast-model")
            >>> system = MemorySystem(
            ...     small_model_fn=my_small_model,
            ...     db_path="project_memories.db",
            ...     max_memories=15
            ... )
        """
        self.storage = MemoryStorage(db_path)
        self.retrieval = MemoryRetrieval(
            small_model_fn=small_model_fn,
            relevance_threshold=relevance_threshold,
            max_memories=max_memories,
            batch_size=batch_size,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay
        )

    # Storage API methods

    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> str:
        """Add a new memory to the system.

        Args:
            text: The content of the memory to store.
            metadata: Optional dictionary of metadata (any JSON-serializable data).
            memory_id: Optional custom ID. If not provided, a UUID is generated.

        Returns:
            The ID of the stored memory.

        Example:
            >>> system = MemorySystem(small_model_fn=lambda x: None)
            >>> mid = system.add_memory(
            ...     "FastAPI is a modern Python web framework",
            ...     metadata={"framework": "fastapi", "language": "python"}
            ... )
        """
        return self.storage.add_memory(text, metadata, memory_id)

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            Memory dictionary with id, text, metadata, and timestamp, or None.

        Example:
            >>> mid = system.add_memory("Example memory")
            >>> memory = system.get_memory(mid)
            >>> memory["text"]
            'Example memory'
        """
        return self.storage.get_memory(memory_id)

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Retrieve all memories from storage.

        Returns:
            List of all memory dictionaries, ordered by timestamp (most recent first).

        Example:
            >>> system.add_memory("Memory 1")
            >>> system.add_memory("Memory 2")
            >>> len(system.get_all_memories())
            2
        """
        return self.storage.get_all_memories()

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory.

        Args:
            memory_id: The ID of the memory to update.
            text: Optional new text content.
            metadata: Optional new metadata (replaces entire metadata dict).

        Returns:
            True if updated successfully, False if memory not found.

        Example:
            >>> mid = system.add_memory("Original")
            >>> system.update_memory(mid, text="Updated")
            True
        """
        return self.storage.update_memory(memory_id, text, metadata)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from the system.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if deleted successfully, False if memory not found.

        Example:
            >>> mid = system.add_memory("Temporary")
            >>> system.delete_memory(mid)
            True
        """
        return self.storage.delete_memory(memory_id)

    def count_memories(self) -> int:
        """Get the total number of memories.

        Returns:
            The count of all memories in storage.

        Example:
            >>> system.add_memory("Memory 1")
            >>> system.add_memory("Memory 2")
            >>> system.count_memories()
            2
        """
        return self.storage.count_memories()

    def clear_all_memories(self) -> int:
        """Delete all memories from the system.

        Returns:
            The number of memories deleted.

        Warning:
            This operation is irreversible.

        Example:
            >>> system.clear_all_memories()
            5
        """
        return self.storage.clear_all()

    # Retrieval API methods

    async def retrieve_relevant_memories(
        self,
        context: str
    ) -> List[ScoredMemory]:
        """Retrieve the most relevant memories for a given context.

        This runs the complete retrieval pipeline:
        1. Loads all memories from storage
        2. Scores each memory using the small model
        3. Filters by relevance threshold
        4. Returns top-K memories

        Args:
            context: The current context/query to find relevant memories for.

        Returns:
            List of ScoredMemory objects, sorted by relevance (highest first).

        Example:
            >>> memories = await system.retrieve_relevant_memories(
            ...     context="How do I handle async operations in Python?"
            ... )
            >>> for mem in memories:
            ...     print(f"Score: {mem.relevance_score:.2f} - {mem.text[:50]}")
        """
        all_memories = self.storage.get_all_memories()
        return await self.retrieval.retrieve_relevant_memories(context, all_memories)

    def format_memories_for_prompt(
        self,
        memories: List[ScoredMemory],
        include_scores: bool = True
    ) -> str:
        """Format memories for inclusion in a prompt.

        Args:
            memories: List of ScoredMemory objects to format.
            include_scores: Whether to include relevance scores. Default: True

        Returns:
            Formatted string ready to include in a prompt.

        Example:
            >>> memories = await system.retrieve_relevant_memories("Python async")
            >>> formatted = system.format_memories_for_prompt(memories)
            >>> print(formatted)
            <memory id="..." score="0.95">
            asyncio is Python's built-in async framework
            </memory>
        """
        return self.retrieval.format_memories_for_prompt(memories, include_scores)

    async def query(
        self,
        context: str,
        task: str,
        primary_model_fn: Callable[[str], Awaitable[str]],
        include_scores: bool = True
    ) -> str:
        """Complete query pipeline: retrieve memories and call primary model.

        This is the high-level API that combines memory retrieval with
        primary model invocation. It:
        1. Retrieves relevant memories based on context
        2. Formats memories for the prompt
        3. Constructs a complete prompt with memories + task
        4. Calls the primary model
        5. Returns the response

        Args:
            context: Current context/situation to find relevant memories for.
            task: The specific task or question for the primary model.
            primary_model_fn: Async function to call the primary LLM.
                Should take a prompt string and return a response string.
            include_scores: Whether to include relevance scores in the prompt.
                Default: True

        Returns:
            The response from the primary model, informed by relevant memories.

        Example:
            >>> async def primary_llm(prompt):
            ...     return await llm_api.complete(prompt, model="gpt-4")
            >>>
            >>> response = await system.query(
            ...     context="User is asking about Python web frameworks",
            ...     task="Compare FastAPI and Flask for a new project",
            ...     primary_model_fn=primary_llm
            ... )
            >>> print(response)
        """
        # Step 1 & 2: Retrieve and format memories
        relevant_memories = await self.retrieve_relevant_memories(context)
        formatted_memories = self.format_memories_for_prompt(
            relevant_memories,
            include_scores
        )

        # Step 3: Construct complete prompt
        prompt = f"""Relevant memories:
{formatted_memories}

Current context: {context}

Task: {task}

Please respond to the task using the relevant memories as context."""

        # Step 4: Call primary model
        response = await primary_model_fn(prompt)

        return response

    async def query_with_custom_prompt(
        self,
        context: str,
        prompt_template: str,
        primary_model_fn: Callable[[str], Awaitable[str]],
        include_scores: bool = True
    ) -> str:
        """Query with a custom prompt template.

        This provides more control over prompt construction. The template
        should include `{memories}` placeholder where memories will be inserted.

        Args:
            context: Current context to find relevant memories for.
            prompt_template: Template string with `{memories}` placeholder.
                Can include other placeholders that will be left as-is.
            primary_model_fn: Async function to call the primary LLM.
            include_scores: Whether to include relevance scores. Default: True

        Returns:
            The response from the primary model.

        Example:
            >>> template = '''
            ... Memories:
            ... {memories}
            ...
            ... Using the above memories, answer: {question}
            ... '''
            >>> response = await system.query_with_custom_prompt(
            ...     context="Python web development",
            ...     prompt_template=template,
            ...     primary_model_fn=primary_llm
            ... )
        """
        # Retrieve and format memories
        relevant_memories = await self.retrieve_relevant_memories(context)
        formatted_memories = self.format_memories_for_prompt(
            relevant_memories,
            include_scores
        )

        # Insert memories into template
        prompt = prompt_template.format(memories=formatted_memories)

        # Call primary model
        response = await primary_model_fn(prompt)

        return response

    # Configuration methods

    def update_retrieval_config(
        self,
        relevance_threshold: Optional[float] = None,
        max_memories: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> None:
        """Update retrieval configuration.

        Args:
            relevance_threshold: New relevance threshold (0-1).
            max_memories: New maximum number of memories to return.
            batch_size: New batch size for parallel API calls.

        Example:
            >>> system.update_retrieval_config(
            ...     relevance_threshold=0.8,
            ...     max_memories=15
            ... )
        """
        if relevance_threshold is not None:
            if not 0 <= relevance_threshold <= 1:
                raise ValueError("relevance_threshold must be between 0 and 1")
            self.retrieval.relevance_threshold = relevance_threshold

        if max_memories is not None:
            if max_memories < 1:
                raise ValueError("max_memories must be at least 1")
            self.retrieval.max_memories = max_memories

        if batch_size is not None:
            if batch_size < 1:
                raise ValueError("batch_size must be at least 1")
            self.retrieval.batch_size = batch_size

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system.

        Returns:
            Dictionary with statistics including:
            - total_memories: Total number of stored memories
            - retrieval_config: Current retrieval configuration

        Example:
            >>> stats = system.get_stats()
            >>> print(f"Total memories: {stats['total_memories']}")
            >>> print(f"Max memories returned: {stats['retrieval_config']['max_memories']}")
        """
        return {
            "total_memories": self.storage.count_memories(),
            "retrieval_config": {
                "relevance_threshold": self.retrieval.relevance_threshold,
                "max_memories": self.retrieval.max_memories,
                "batch_size": self.retrieval.batch_size,
                "retry_attempts": self.retrieval.retry_attempts,
                "retry_delay": self.retrieval.retry_delay
            }
        }
