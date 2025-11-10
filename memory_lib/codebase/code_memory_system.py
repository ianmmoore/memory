"""Main API for code memory system.

This module provides the high-level CodeMemorySystem class that integrates
code-specific storage, indexing, and retrieval into a unified API for
code intelligence applications.
"""

from typing import Dict, List, Optional, Any, Callable, Awaitable
from pathlib import Path

from .code_storage import CodeMemoryStorage
from .code_retrieval import CodeMemoryRetrieval, CodeContext, ScoredMemory
from .indexer import CodeIndexer


class CodeMemorySystem:
    """Complete code memory system with storage, indexing, and intelligent retrieval.

    This class provides a comprehensive solution for code intelligence:
    - Indexes code repositories to extract functions, classes, and documentation
    - Stores code memories with specialized metadata
    - Uses LLM-based retrieval with caching and optimization
    - Provides easy-to-use API for querying with code context

    Attributes:
        storage: CodeMemoryStorage instance for persistence.
        retrieval: CodeMemoryRetrieval instance for intelligent retrieval.
        indexer: CodeIndexer instance for extracting code entities.

    Example:
        >>> async def small_model_api(prompt: str) -> str:
        ...     # Call small LLM for scoring
        ...     return await llm_client.complete(prompt, model="gpt-3.5-turbo")
        >>>
        >>> async def primary_model_api(prompt: str) -> str:
        ...     # Call primary LLM for main tasks
        ...     return await llm_client.complete(prompt, model="gpt-4")
        >>>
        >>> # Initialize system
        >>> code_memory = CodeMemorySystem(
        ...     small_model_fn=small_model_api,
        ...     db_path="project_memories.db"
        ... )
        >>>
        >>> # Index a codebase
        >>> code_memory.index_repository(
        ...     "src/",
        ...     exclude_patterns=["*/tests/*", "*/__pycache__/*"]
        ... )
        >>>
        >>> # Query with code context
        >>> context = CodeContext(
        ...     user_query="Fix authentication bug in login handler",
        ...     current_file="api/auth.py",
        ...     errors="AttributeError: 'NoneType' object has no attribute 'token'"
        ... )
        >>> response = await code_memory.query(
        ...     context=context,
        ...     primary_model_fn=primary_model_api
        ... )
    """

    def __init__(
        self,
        small_model_fn: Callable[[str], Awaitable[str]],
        db_path: str = "code_memories.db",
        relevance_threshold: float = 0.7,
        max_memories: int = 15,
        batch_size: int = 10,
        enable_caching: bool = True,
        enable_dependency_boost: bool = True,
        enable_recency_boost: bool = True
    ):
        """Initialize the code memory system.

        Args:
            small_model_fn: Async function for calling small LLM to score relevance.
            db_path: Path to SQLite database for persistent storage.
                Default: "code_memories.db"
            relevance_threshold: Minimum relevance score (0-1) for inclusion.
                Default: 0.7
            max_memories: Maximum number of memories to return (K).
                Default: 15 (higher for code)
            batch_size: Number of parallel API calls for scoring.
                Default: 10
            enable_caching: Whether to cache relevance scores.
                Default: True
            enable_dependency_boost: Whether to boost scores for dependencies.
                Default: True
            enable_recency_boost: Whether to boost recently modified files.
                Default: True

        Example:
            >>> async def my_small_model(prompt):
            ...     return await api.complete(prompt)
            >>> system = CodeMemorySystem(
            ...     small_model_fn=my_small_model,
            ...     db_path="my_project.db",
            ...     max_memories=20
            ... )
        """
        self.storage = CodeMemoryStorage(db_path)
        self.retrieval = CodeMemoryRetrieval(
            small_model_fn=small_model_fn,
            relevance_threshold=relevance_threshold,
            max_memories=max_memories,
            batch_size=batch_size,
            enable_caching=enable_caching,
            enable_dependency_boost=enable_dependency_boost,
            enable_recency_boost=enable_recency_boost
        )
        self.indexer = CodeIndexer()

    # Indexing API

    def index_file(self, file_path: str, auto_store: bool = True) -> List[Dict[str, Any]]:
        """Index a single source file and optionally store memories.

        Args:
            file_path: Path to the source file to index.
            auto_store: If True, automatically store extracted entities as memories.
                Default: True

        Returns:
            List of extracted entity dictionaries.

        Example:
            >>> entities = system.index_file("src/utils.py")
            >>> print(f"Indexed {len(entities)} entities")
        """
        entities = self.indexer.index_file(file_path)

        if auto_store:
            for entity in entities:
                self.storage.add_code_memory(**entity)

        return entities

    def index_repository(
        self,
        directory: str,
        exclude_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        auto_store: bool = True
    ) -> List[Dict[str, Any]]:
        """Index an entire code repository.

        Args:
            directory: Path to the repository directory.
            exclude_patterns: List of glob patterns to exclude
                (e.g., ["*/tests/*", "*/__pycache__/*", "*.pyc"]).
            recursive: Whether to recursively index subdirectories.
                Default: True
            auto_store: If True, automatically store all extracted entities.
                Default: True

        Returns:
            List of all extracted entity dictionaries.

        Example:
            >>> entities = system.index_repository(
            ...     "src/",
            ...     exclude_patterns=["*/tests/*", "*/.venv/*"],
            ...     recursive=True
            ... )
            >>> print(f"Indexed {len(entities)} entities from repository")
        """
        entities = self.indexer.index_directory(
            directory,
            exclude_patterns=exclude_patterns,
            recursive=recursive
        )

        if auto_store:
            for entity in entities:
                self.storage.add_code_memory(**entity)

        return entities

    def reindex_file(self, file_path: str) -> int:
        """Re-index a file after modifications.

        This removes old memories for the file and indexes it again.

        Args:
            file_path: Path to the file to re-index.

        Returns:
            Number of new entities indexed.

        Example:
            >>> # After editing a file
            >>> count = system.reindex_file("src/api/handlers.py")
            >>> print(f"Re-indexed {count} entities")
        """
        # Remove old memories for this file
        old_memories = self.storage.get_memories_by_file(file_path)
        for mem in old_memories:
            self.storage.delete_memory(mem["id"])

        # Invalidate cache for this file
        self.retrieval.invalidate_cache_for_file(file_path)

        # Index and store new entities
        entities = self.index_file(file_path, auto_store=True)

        return len(entities)

    # Storage API

    def add_code_memory(self, **kwargs) -> str:
        """Add a code memory manually.

        Args:
            **kwargs: Code memory fields (file_path, entity_name, code_snippet, etc.)

        Returns:
            Memory ID.

        Example:
            >>> mid = system.add_code_memory(
            ...     file_path="api/utils.py",
            ...     entity_name="helper_function",
            ...     code_snippet="def helper(): pass",
            ...     language="python"
            ... )
        """
        return self.storage.add_code_memory(**kwargs)

    def add_documentation_memory(
        self,
        title: str,
        content: str,
        category: str = "documentation",
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a documentation memory (README, architecture doc, etc.).

        Args:
            title: Title of the documentation.
            content: Full documentation content.
            category: Category (e.g., "readme", "architecture", "api-docs").
                Default: "documentation"
            file_path: Optional associated file path.
            metadata: Additional metadata.

        Returns:
            Memory ID.

        Example:
            >>> mid = system.add_documentation_memory(
            ...     title="API Architecture",
            ...     content="Our API uses a layered architecture...",
            ...     category="architecture"
            ... )
        """
        return self.storage.add_non_code_memory(
            category=category,
            title=title,
            content=content,
            file_path=file_path,
            metadata=metadata
        )

    def add_debugging_session(
        self,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a debugging session memory.

        Args:
            title: Brief description of the bug/issue.
            content: Detailed description of the issue and resolution.
            metadata: Additional metadata (e.g., severity, date, files affected).

        Returns:
            Memory ID.

        Example:
            >>> mid = system.add_debugging_session(
            ...     title="Fixed race condition in async handler",
            ...     content="Issue: Multiple concurrent requests...",
            ...     metadata={"severity": "high", "files": ["api/handlers.py"]}
            ... )
        """
        return self.storage.add_non_code_memory(
            category="debugging",
            title=title,
            content=content,
            metadata=metadata
        )

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID.

        Args:
            memory_id: Memory ID.

        Returns:
            Memory dictionary or None.
        """
        # Try code memories first
        mem = self.storage.get_code_memory(memory_id)
        if mem:
            return mem

        # Try non-code memories
        return self.storage.get_non_code_memory(memory_id)

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories (code and non-code).

        Returns:
            List of all memory dictionaries.
        """
        return self.storage.get_all_memories()

    def get_memories_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all memories for a specific file.

        Args:
            file_path: Path to the file.

        Returns:
            List of memory dictionaries.
        """
        return self.storage.get_memories_by_file(file_path)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: Memory ID.

        Returns:
            True if deleted, False if not found.
        """
        return self.storage.delete_memory(memory_id)

    # Retrieval API

    async def retrieve_relevant_memories(
        self,
        context: CodeContext,
        file_hashes: Optional[Dict[str, str]] = None
    ) -> List[ScoredMemory]:
        """Retrieve relevant memories for a code context.

        Args:
            context: CodeContext with user query and current coding situation.
            file_hashes: Optional dictionary of file_path -> hash for caching.

        Returns:
            List of ScoredMemory objects, sorted by relevance.

        Example:
            >>> context = CodeContext(
            ...     user_query="Implement user login",
            ...     current_file="auth.py"
            ... )
            >>> memories = await system.retrieve_relevant_memories(context)
            >>> for mem in memories:
            ...     print(f"{mem.relevance_score:.2f}: {mem.text}")
        """
        all_memories = self.storage.get_all_memories()
        return await self.retrieval.retrieve_code_memories(
            context,
            all_memories,
            file_hashes
        )

    def format_memories_for_prompt(
        self,
        memories: List[ScoredMemory],
        group_by_file: bool = True,
        include_scores: bool = True
    ) -> str:
        """Format memories for inclusion in a prompt.

        Args:
            memories: List of ScoredMemory objects.
            group_by_file: Whether to group entities from same file.
                Default: True
            include_scores: Whether to include relevance scores.
                Default: True

        Returns:
            Formatted string.
        """
        return self.retrieval.format_code_memories_for_prompt(
            memories,
            group_by_file,
            include_scores
        )

    async def query(
        self,
        context: CodeContext,
        primary_model_fn: Callable[[str], Awaitable[str]],
        include_scores: bool = True,
        file_hashes: Optional[Dict[str, str]] = None
    ) -> str:
        """Complete query pipeline: retrieve memories and call primary model.

        This is the high-level API for querying with code context.

        Args:
            context: CodeContext describing the current situation and task.
            primary_model_fn: Async function to call primary LLM.
            include_scores: Whether to include scores in prompt.
                Default: True
            file_hashes: Optional file hashes for caching.

        Returns:
            Response from the primary model.

        Example:
            >>> async def primary_llm(prompt):
            ...     return await api.complete(prompt, model="gpt-4")
            >>>
            >>> context = CodeContext(
            ...     user_query="Add error handling to the login function",
            ...     current_file="auth.py",
            ...     errors="Unhandled exception in validate_credentials"
            ... )
            >>> response = await system.query(context, primary_llm)
            >>> print(response)
        """
        # Retrieve relevant memories
        relevant_memories = await self.retrieve_relevant_memories(context, file_hashes)

        # Format memories
        formatted_memories = self.format_memories_for_prompt(
            relevant_memories,
            include_scores=include_scores
        )

        # Construct prompt
        prompt = f"""Relevant code context:
{formatted_memories}

Current situation:
{context.to_context_string()}

Please help with the task using the relevant code context provided above."""

        # Call primary model
        response = await primary_model_fn(prompt)

        return response

    async def query_with_custom_prompt(
        self,
        context: CodeContext,
        prompt_template: str,
        primary_model_fn: Callable[[str], Awaitable[str]],
        include_scores: bool = True,
        file_hashes: Optional[Dict[str, str]] = None
    ) -> str:
        """Query with a custom prompt template.

        Args:
            context: CodeContext for retrieval.
            prompt_template: Template with {memories} and {context} placeholders.
            primary_model_fn: Async function to call primary LLM.
            include_scores: Whether to include scores.
            file_hashes: Optional file hashes.

        Returns:
            Response from primary model.

        Example:
            >>> template = '''
            ... Code context:
            ... {memories}
            ...
            ... Task: {context}
            ...
            ... Provide a detailed implementation plan.
            ... '''
            >>> response = await system.query_with_custom_prompt(
            ...     context,
            ...     template,
            ...     primary_llm
            ... )
        """
        # Retrieve and format memories
        relevant_memories = await self.retrieve_relevant_memories(context, file_hashes)
        formatted_memories = self.format_memories_for_prompt(
            relevant_memories,
            include_scores=include_scores
        )

        # Fill template
        prompt = prompt_template.format(
            memories=formatted_memories,
            context=context.to_context_string()
        )

        # Call primary model
        response = await primary_model_fn(prompt)

        return response

    # Configuration and stats

    def update_retrieval_config(
        self,
        relevance_threshold: Optional[float] = None,
        max_memories: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> None:
        """Update retrieval configuration.

        Args:
            relevance_threshold: New threshold (0-1).
            max_memories: New maximum memories to return.
            batch_size: New batch size for parallel API calls.
        """
        if relevance_threshold is not None:
            self.retrieval.relevance_threshold = relevance_threshold

        if max_memories is not None:
            self.retrieval.max_memories = max_memories

        if batch_size is not None:
            self.retrieval.batch_size = batch_size

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics.

        Returns:
            Dictionary with statistics about memories, cache, and configuration.

        Example:
            >>> stats = system.get_stats()
            >>> print(f"Total code memories: {stats['code_memories']}")
            >>> print(f"Cache size: {stats['cache']['cache_size']}")
        """
        return {
            "code_memories": self.storage.count_code_memories(),
            "non_code_memories": self.storage.count_non_code_memories(),
            "total_memories": self.storage.count_memories(),
            "retrieval_config": {
                "relevance_threshold": self.retrieval.relevance_threshold,
                "max_memories": self.retrieval.max_memories,
                "batch_size": self.retrieval.batch_size
            },
            "cache": self.retrieval.get_cache_stats()
        }

    def clear_cache(self) -> None:
        """Clear the relevance score cache."""
        if self.retrieval.cache:
            self.retrieval.cache.clear()

    def clear_all_memories(self) -> int:
        """Delete all memories from the system.

        Returns:
            Number of memories deleted.

        Warning:
            This is irreversible.
        """
        return self.storage.clear_all()
