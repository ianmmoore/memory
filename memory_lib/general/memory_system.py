"""Main memory system API combining storage and retrieval.

This module provides the high-level MemorySystem class that integrates storage
and LLM-based retrieval into a unified, easy-to-use API.

Supports optional embedding-based prefiltering for cost reduction at scale.
"""

from typing import Dict, List, Optional, Any, Callable, Awaitable
from .storage import MemoryStorage
from .retrieval import MemoryRetrieval, ScoredMemory
from .prefilter import EmbeddingPrefilter, EmbeddingConfig, create_openai_embedding_fn


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
        retry_delay: float = 1.0,
        embedding_fn: Optional[Callable[[List[str]], Awaitable[List[List[float]]]]] = None,
        enable_prefilter: bool = False,
        prefilter_top_k: int = 100
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
            embedding_fn: Optional async function for generating embeddings.
                Takes a list of texts and returns a list of embedding vectors.
                Required if enable_prefilter is True.
            enable_prefilter: Whether to enable embedding-based prefiltering.
                When True, only top prefilter_top_k candidates by embedding similarity
                are scored by the LLM, reducing costs. Default: False
            prefilter_top_k: Number of candidates to select via prefiltering.
                Only used when enable_prefilter is True. Default: 100

        Example:
            >>> async def my_small_model(prompt):
            ...     return await llm_api.complete(prompt, model="fast-model")
            >>> system = MemorySystem(
            ...     small_model_fn=my_small_model,
            ...     db_path="project_memories.db",
            ...     max_memories=15
            ... )

        Example with prefiltering:
            >>> from memory_lib.general.prefilter import create_openai_embedding_fn
            >>> embed_fn = create_openai_embedding_fn(api_key="sk-...")
            >>> system = MemorySystem(
            ...     small_model_fn=my_small_model,
            ...     embedding_fn=embed_fn,
            ...     enable_prefilter=True,
            ...     prefilter_top_k=100
            ... )
        """
        self.storage = MemoryStorage(db_path)
        self.embedding_fn = embedding_fn
        self.prefilter_enabled = enable_prefilter

        # Set up prefilter if enabled
        prefilter = None
        if enable_prefilter and embedding_fn is not None:
            prefilter = EmbeddingPrefilter(embedding_fn=embedding_fn)

        self.retrieval = MemoryRetrieval(
            small_model_fn=small_model_fn,
            relevance_threshold=relevance_threshold,
            max_memories=max_memories,
            batch_size=batch_size,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            prefilter=prefilter,
            prefilter_top_k=prefilter_top_k
        )

    # Storage API methods

    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        """Add a new memory to the system.

        Args:
            text: The content of the memory to store.
            metadata: Optional dictionary of metadata (any JSON-serializable data).
            memory_id: Optional custom ID. If not provided, a UUID is generated.
            embedding: Optional pre-computed embedding vector. If None and prefiltering
                is enabled, you can generate embeddings later with generate_embeddings().

        Returns:
            The ID of the stored memory.

        Example:
            >>> system = MemorySystem(small_model_fn=lambda x: None)
            >>> mid = system.add_memory(
            ...     "FastAPI is a modern Python web framework",
            ...     metadata={"framework": "fastapi", "language": "python"}
            ... )
        """
        return self.storage.add_memory(text, metadata, memory_id, embedding)

    async def add_memory_with_embedding(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> str:
        """Add a memory and automatically generate its embedding.

        This is a convenience method that generates the embedding for you.
        Requires embedding_fn to be configured.

        Args:
            text: The content of the memory to store.
            metadata: Optional dictionary of metadata.
            memory_id: Optional custom ID.

        Returns:
            The ID of the stored memory.

        Raises:
            ValueError: If embedding_fn is not configured.
        """
        if self.embedding_fn is None:
            raise ValueError("embedding_fn is not configured. Cannot generate embedding.")

        embeddings = await self.embedding_fn([text])
        embedding = embeddings[0]

        return self.storage.add_memory(text, metadata, memory_id, embedding)

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
            - memories_with_embeddings: Count of memories that have embeddings
            - retrieval_config: Current retrieval configuration
            - prefilter_enabled: Whether prefiltering is active

        Example:
            >>> stats = system.get_stats()
            >>> print(f"Total memories: {stats['total_memories']}")
            >>> print(f"Max memories returned: {stats['retrieval_config']['max_memories']}")
        """
        total = self.storage.count_memories()
        without_embeddings = len(self.storage.get_memories_without_embeddings())

        return {
            "total_memories": total,
            "memories_with_embeddings": total - without_embeddings,
            "memories_without_embeddings": without_embeddings,
            "prefilter_enabled": self.prefilter_enabled and self.retrieval.prefilter is not None,
            "prefilter_top_k": self.retrieval.prefilter_top_k,
            "retrieval_config": {
                "relevance_threshold": self.retrieval.relevance_threshold,
                "max_memories": self.retrieval.max_memories,
                "batch_size": self.retrieval.batch_size,
                "retry_attempts": self.retrieval.retry_attempts,
                "retry_delay": self.retrieval.retry_delay
            }
        }

    # Embedding and prefilter methods

    async def generate_embeddings(
        self,
        batch_size: int = 100,
        force: bool = False
    ) -> int:
        """Generate embeddings for memories that don't have them.

        This method batch-processes memories to generate their embeddings,
        enabling prefiltering for faster retrieval. Uses pagination to avoid
        loading all memories into memory at once.

        Args:
            batch_size: Number of memories to embed in each batch. Default: 100
            force: If True, regenerate embeddings for all memories.
                If False, only generate for memories without embeddings. Default: False

        Returns:
            Number of embeddings generated.

        Raises:
            ValueError: If embedding_fn is not configured.

        Example:
            >>> count = await system.generate_embeddings(batch_size=50)
            >>> print(f"Generated {count} embeddings")
        """
        import logging
        import gc
        logger = logging.getLogger(__name__)

        if self.embedding_fn is None:
            raise ValueError("embedding_fn is not configured. Cannot generate embeddings.")

        if force:
            # For force mode, still load all memories (less common use case)
            memories = self.storage.get_all_memories(include_embeddings=False)
            total = len(memories)

            generated = 0
            for i in range(0, len(memories), batch_size):
                batch = memories[i:i + batch_size]
                texts = [m["text"] for m in batch]
                embeddings = await self.embedding_fn(texts)

                for memory, embedding in zip(batch, embeddings):
                    self.storage.update_embedding(memory["id"], embedding)
                    generated += 1

                logger.info(f"Embedding progress: {generated}/{total} ({100*generated/total:.1f}%)")

            return generated

        # Paginated approach: query batch_size memories at a time
        total = self.storage.count_memories_without_embeddings()
        if total == 0:
            return 0

        logger.info(f"Generating embeddings for {total} memories in batches of {batch_size}")
        generated = 0

        while True:
            # Always get the first batch_size memories without embeddings
            # (offset=0 since we're embedding them and they won't appear in next query)
            batch = self.storage.get_memories_without_embeddings(limit=batch_size, offset=0)

            if not batch:
                break

            texts = [m["text"] for m in batch]
            embeddings = await self.embedding_fn(texts)

            for memory, embedding in zip(batch, embeddings):
                self.storage.update_embedding(memory["id"], embedding)
                generated += 1

            logger.info(f"Embedding progress: {generated}/{total} ({100*generated/total:.1f}%)")

            # Explicit cleanup to help with memory
            del batch, texts, embeddings
            gc.collect()

        return generated

    def enable_prefilter(
        self,
        top_k: int = 100,
        embedding_fn: Optional[Callable[[List[str]], Awaitable[List[List[float]]]]] = None
    ) -> None:
        """Enable embedding-based prefiltering.

        This reduces LLM API costs by first selecting candidates based on
        embedding similarity, then only scoring those candidates.

        Args:
            top_k: Number of candidates to select before LLM scoring. Default: 100
            embedding_fn: Optional embedding function. If not provided, uses the
                one configured at initialization.

        Raises:
            ValueError: If no embedding function is available.

        Example:
            >>> system.enable_prefilter(top_k=50)
            >>> # Now retrieval will only score top 50 candidates by embedding similarity
        """
        if embedding_fn is not None:
            self.embedding_fn = embedding_fn

        if self.embedding_fn is None:
            raise ValueError("No embedding function available. Provide one via enable_prefilter() or at initialization.")

        prefilter = EmbeddingPrefilter(embedding_fn=self.embedding_fn)
        self.retrieval.set_prefilter(prefilter, top_k)
        self.prefilter_enabled = True

    def disable_prefilter(self) -> None:
        """Disable embedding-based prefiltering.

        After disabling, all memories will be scored by the LLM (exhaustive mode).
        """
        self.retrieval.set_prefilter(None)
        self.prefilter_enabled = False

    def update_prefilter_config(self, top_k: int) -> None:
        """Update the number of candidates for prefiltering.

        Args:
            top_k: New number of candidates to select before LLM scoring.
        """
        self.retrieval.prefilter_top_k = top_k

    # =========================================================================
    # Extraction and Update Methods (using primary_model_fn / gpt-5.1)
    # =========================================================================

    async def extract_memories_from_dialogue(
        self,
        dialogue: str,
        primary_model_fn: Callable[[str], Awaitable[str]],
        auto_store: bool = True
    ) -> List[str]:
        """Extract memories from a conversation using the primary model.

        This uses gpt-5.1 (or configured primary model) to extract
        important facts, preferences, and information from a dialogue.

        Args:
            dialogue: The conversation text to extract memories from.
            primary_model_fn: Async function to call the primary LLM (gpt-5.1).
            auto_store: If True, automatically store extracted memories.
                Default: True

        Returns:
            List of extracted memory strings.

        Example:
            >>> dialogue = '''
            ... User: My name is Alice and I'm a software engineer.
            ... Assistant: Nice to meet you! What languages do you use?
            ... User: Mainly Python and Rust.
            ... '''
            >>> memories = await system.extract_memories_from_dialogue(
            ...     dialogue, primary_model_fn=gpt51_fn
            ... )
            >>> print(memories)
            ["User's name is Alice", "User is a software engineer",
             "User mainly uses Python and Rust"]
        """
        prompt = f"""Extract important facts, preferences, and personal information from this conversation.
Return each memory as a separate line. Be concise - each memory should be one clear statement.
Focus on facts about the user that would be useful to remember for future conversations.

CONVERSATION:
{dialogue}

MEMORIES (one per line):"""

        response = await primary_model_fn(prompt)

        # Parse response into individual memories
        memories = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove common prefixes like "- ", "* ", "1. "
            if line.startswith(('-', '*', '•')):
                line = line[1:].strip()
            elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
                line = line[2:].strip()
            if line:
                memories.append(line)

        # Optionally store the memories
        if auto_store:
            for mem in memories:
                self.add_memory(mem, metadata={"source": "extraction"})

        return memories

    async def decide_memory_update(
        self,
        existing_memory: str,
        new_information: str,
        primary_model_fn: Callable[[str], Awaitable[str]]
    ) -> Dict[str, str]:
        """Decide how to handle potentially conflicting information.

        Uses the primary model (gpt-5.1) to determine the appropriate
        action when new information may conflict with existing memories.

        Args:
            existing_memory: The current memory text.
            new_information: New information that may conflict.
            primary_model_fn: Async function to call the primary LLM (gpt-5.1).

        Returns:
            Dict with keys:
                - action: "UPDATE", "ADD", "DELETE", or "NOOP"
                - result: New memory text (if UPDATE or ADD)
                - reasoning: Brief explanation

        Example:
            >>> decision = await system.decide_memory_update(
            ...     existing_memory="User has 2 cats",
            ...     new_information="User mentioned they now have 3 cats",
            ...     primary_model_fn=gpt51_fn
            ... )
            >>> print(decision)
            {"action": "UPDATE", "result": "User has 3 cats",
             "reasoning": "Number of cats increased"}
        """
        prompt = f"""You are a memory management system. Given an existing memory and new information, decide what action to take.

EXISTING MEMORY:
{existing_memory}

NEW INFORMATION:
{new_information}

Decide:
- UPDATE: New info contradicts/refines the existing memory
- ADD: New info is additional, doesn't replace existing
- DELETE: Existing memory is now obsolete/wrong
- NOOP: No change needed (already captured or irrelevant)

Respond in exactly this format:
ACTION: <UPDATE|ADD|DELETE|NOOP>
RESULT: <new memory text if UPDATE/ADD, otherwise N/A>
REASONING: <brief explanation>"""

        response = await primary_model_fn(prompt)

        # Parse response
        result = {"action": "NOOP", "result": "", "reasoning": ""}
        for line in response.strip().split('\n'):
            if line.startswith("ACTION:"):
                action = line.split(":", 1)[1].strip().upper()
                if action in ("UPDATE", "ADD", "DELETE", "NOOP"):
                    result["action"] = action
            elif line.startswith("RESULT:"):
                result["result"] = line.split(":", 1)[1].strip()
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()

        return result

    async def process_new_information(
        self,
        new_info: str,
        primary_model_fn: Callable[[str], Awaitable[str]],
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Process new information and automatically handle conflicts.

        This is the main entry point for adding information that may
        conflict with existing memories. It:
        1. Finds potentially related memories (using gpt-5-nano for scoring)
        2. Checks for conflicts with each (using gpt-5.1 for decisions)
        3. Executes appropriate updates

        Args:
            new_info: New information to process.
            primary_model_fn: Async function to call the primary LLM (gpt-5.1).
            similarity_threshold: Threshold for finding related memories.

        Returns:
            Dict with processing results including actions taken.

        Example:
            >>> result = await system.process_new_information(
            ...     "User now lives in San Francisco",
            ...     primary_model_fn=gpt51_fn
            ... )
            >>> print(result)
            {"actions_taken": [{"action": "UPDATE", "old": "User lives in NYC",
                               "new": "User lives in San Francisco"}]}
        """
        # Find potentially related memories (uses small model for scoring)
        related = await self.retrieve_relevant_memories(
            new_info,
            threshold_override=similarity_threshold
        )

        actions_taken = []

        if not related:
            # No related memories - just add as new
            mem_id = self.add_memory(new_info, metadata={"source": "processed"})
            actions_taken.append({
                "action": "ADD",
                "memory_id": mem_id,
                "text": new_info,
                "reason": "No related memories found"
            })
        else:
            # Check each related memory for conflicts (uses primary model)
            for scored_mem in related:
                decision = await self.decide_memory_update(
                    existing_memory=scored_mem.text,
                    new_information=new_info,
                    primary_model_fn=primary_model_fn
                )

                if decision["action"] == "UPDATE":
                    self.update_memory(scored_mem.memory_id, text=decision["result"])
                    actions_taken.append({
                        "action": "UPDATE",
                        "memory_id": scored_mem.memory_id,
                        "old_text": scored_mem.text,
                        "new_text": decision["result"],
                        "reason": decision["reasoning"]
                    })
                elif decision["action"] == "DELETE":
                    self.delete_memory(scored_mem.memory_id)
                    actions_taken.append({
                        "action": "DELETE",
                        "memory_id": scored_mem.memory_id,
                        "text": scored_mem.text,
                        "reason": decision["reasoning"]
                    })
                elif decision["action"] == "ADD":
                    mem_id = self.add_memory(
                        decision["result"],
                        metadata={"source": "processed", "related_to": scored_mem.memory_id}
                    )
                    actions_taken.append({
                        "action": "ADD",
                        "memory_id": mem_id,
                        "text": decision["result"],
                        "reason": decision["reasoning"]
                    })
                # NOOP - do nothing

        return {"actions_taken": actions_taken}
