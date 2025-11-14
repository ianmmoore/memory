"""Memory-guided Terminus agent for Terminal-Bench."""

import os
from pathlib import Path
from typing import Optional
import litellm
import logging

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.llms.chat import Chat
from harbor.models.agent.name import AgentName

# Import custom ResponsesLLM for GPT-5 models
from terminal_bench_agent.responses_llm import ResponsesLLM

# Import cleanup managers
from terminal_bench_agent.cleanup_manager import DaytonaCleanupManager, DockerCleanupManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import memory system
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "memory_lib"))
    from memory_lib import CodeMemorySystem
    from memory_lib.codebase import CodeContext
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    CodeMemorySystem = None
    CodeContext = None


class MemoryGuidedTerminus(Terminus2):
    """Terminus2 agent enhanced with CodeMemorySystem for code intelligence.

    This agent extends the proven Terminus2 framework with advanced code memory capabilities:
    - Indexes and understands code structure (functions, classes, imports)
    - Retrieves relevant code using dependency-aware and recency-weighted retrieval
    - Provides rich code context with smart caching and boosting
    - Injects code-aware memory context into LLM prompts
    - Uses all Terminus2 features: tmux sessions, summarization, ATIF tracking
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        memory_model: str | None = None,
        enable_memory: bool = True,
        max_turns: int | None = None,
        parser_name: str = "json",
        api_base: str | None = None,
        temperature: float = 0.7,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        session_id: str | None = None,
        enable_summarize: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize memory-guided Terminus agent.

        Args:
            logs_dir: Directory for logs
            model_name: Main LLM model (e.g., "gpt-5-codex")
            memory_model: Model for memory retrieval (e.g., "gpt-5-nano")
            enable_memory: Whether to use memory system
            max_turns: Maximum conversation turns
            parser_name: Response parser ("json" or "xml")
            api_base: Optional API base URL
            temperature: LLM temperature
            logprobs: Whether to request log probabilities
            top_logprobs: Number of top log probs to return
            session_id: Optional session ID
            enable_summarize: Whether to enable summarization
        """
        # gpt-5-codex requires temperature=1.0
        if model_name and "gpt-5" in model_name.lower():
            temperature = 1.0

        # Initialize parent Terminus2
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            max_turns=max_turns,
            parser_name=parser_name,
            api_base=api_base,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            session_id=session_id,
            enable_summarize=enable_summarize,
            *args,
            **kwargs,
        )

        # Override LLM with ResponsesLLM to support GPT-5 models
        self._llm = ResponsesLLM(
            model_name=model_name,
            api_base=api_base,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            session_id=session_id,
        )

        # Initialize memory system
        self._enable_memory = enable_memory and MEMORY_AVAILABLE
        self._memory_model = memory_model or os.environ.get("MEMORY_MODEL", "gpt-5-nano")
        self.memory = None

        if self._enable_memory:
            try:
                # Create async LLM wrapper for memory retrieval
                async def memory_llm_fn(prompt: str) -> str:
                    """Wrapper for calling LiteLLM for memory retrieval scoring."""
                    # Check if memory model needs Responses API (GPT-5 models)
                    model_lower = self._memory_model.lower()
                    uses_responses_api = "gpt-5" in model_lower or "o3" in model_lower or "o1" in model_lower

                    if uses_responses_api:
                        # Use Responses API for GPT-5 models
                        response = await litellm.aresponses(
                            model=self._memory_model,
                            input=f"user: {prompt}",
                            temperature=0.0,  # Deterministic for relevance scoring
                        )
                        # Extract output from Responses API response
                        if hasattr(response, 'output'):
                            return response.output
                        elif isinstance(response, dict) and 'output' in response:
                            return response['output']
                        else:
                            return str(response)
                    else:
                        # Use standard completion API for non-GPT-5 models
                        response = await litellm.acompletion(
                            model=self._memory_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,  # Deterministic for relevance scoring
                        )
                        return response.choices[0].message.content

                # Initialize CodeMemorySystem with code-aware storage
                memory_db_path = logs_dir / "code_memories.db"
                self.memory = CodeMemorySystem(
                    small_model_fn=memory_llm_fn,
                    db_path=str(memory_db_path),
                    relevance_threshold=0.7,  # Higher threshold for code precision
                    max_memories=15,  # More memories needed for code context
                    enable_caching=True,  # Cache relevance scores
                    enable_dependency_boost=True,  # Boost related files
                    enable_recency_boost=True,  # Prioritize recent changes
                )
                self._logger.info(f"Memory system enabled with {self._memory_model} (db: {memory_db_path})")
            except Exception as e:
                self._logger.warning(f"Failed to initialize memory system: {e}")
                self._enable_memory = False
                self.memory = None
        else:
            if not MEMORY_AVAILABLE:
                self._logger.info("Memory system not available - memory_lib not found")
            else:
                self._logger.info("Memory system disabled by configuration")

        # Initialize cleanup managers
        self.daytona_cleanup = None
        self.docker_cleanup = DockerCleanupManager()

        # Try to initialize Daytona cleanup if available
        try:
            if os.environ.get("DAYTONA_API_KEY"):
                self.daytona_cleanup = DaytonaCleanupManager()
                self._logger.info("Daytona cleanup manager initialized")
        except Exception as e:
            self._logger.warning(f"Could not initialize Daytona cleanup: {e}")
            self.daytona_cleanup = None

    @staticmethod
    def name() -> str:
        """Return agent name."""
        return "terminus-2"

    @staticmethod
    def version() -> str:
        """Return agent version."""
        return "1.0.0"

    async def _index_environment(self, working_dir: str = "/workspace") -> int:
        """Index the environment's codebase for code memory retrieval.

        Args:
            working_dir: Directory to index (default: /workspace for Docker)

        Returns:
            Number of entities indexed
        """
        if not self.memory or not self._enable_memory:
            return 0

        try:
            # Index the workspace directory, excluding common non-code paths
            exclude_patterns = [
                "*/.git/*",
                "*/__pycache__/*",
                "*/node_modules/*",
                "*/venv/*",
                "*/.venv/*",
                "*/dist/*",
                "*/build/*",
                "*.pyc",
                "*.pyo",
                "*.so",
                "*.dylib",
            ]

            self._logger.info(f"Indexing environment: {working_dir}")
            entities = self.memory.index_repository(
                working_dir,
                exclude_patterns=exclude_patterns,
                recursive=True,
                auto_store=True
            )

            self._logger.info(f"Indexed {len(entities)} code entities from environment")
            return len(entities)

        except Exception as e:
            self._logger.warning(f"Failed to index environment: {e}", exc_info=True)
            return 0

    async def _query_llm(
        self,
        chat: Chat,
        prompt: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> str:
        """Query LLM with memory-enhanced prompt.

        This method overrides Terminus2's _query_llm to inject relevant
        memory context before querying the LLM.

        Args:
            chat: Chat conversation object
            prompt: Prompt to send to LLM
            logging_paths: Paths for logging
            original_instruction: Original task instruction
            session: Optional tmux session

        Returns:
            LLM response string
        """
        # Inject memory context if available
        enhanced_prompt = await self._enhance_prompt_with_memory(
            prompt, original_instruction
        )

        # Call parent's _query_llm with enhanced prompt
        return await super()._query_llm(
            chat=chat,
            prompt=enhanced_prompt,
            logging_paths=logging_paths,
            original_instruction=original_instruction,
            session=session,
        )

    async def _enhance_prompt_with_memory(
        self, prompt: str, instruction: str
    ) -> str:
        """Enhance prompt with relevant memory context.

        Args:
            prompt: Original prompt
            instruction: Task instruction

        Returns:
            Enhanced prompt with memory context
        """
        if not self.memory or not instruction:
            return prompt

        try:
            # Retrieve relevant memories for this task
            # Use the instruction as the query for memory retrieval
            relevant_context = await self._retrieve_memory_context(instruction)

            if relevant_context:
                # Inject memory context into the prompt
                # Add it before the current prompt to provide context
                enhanced = f"""{prompt}

## RELEVANT EXPERIENCE FROM SIMILAR TASKS

You have previously worked on similar terminal-based coding tasks. Here is relevant context that may help:

{relevant_context}

Use this experience to inform your approach, but adapt as needed for the current task.

---

Now, continue with your response:"""

                self._logger.debug(f"Enhanced prompt with {len(relevant_context)} chars of memory context")
                return enhanced
            else:
                return prompt

        except Exception as e:
            self._logger.warning(f"Failed to enhance prompt with memory: {e}")
            return prompt

    async def _retrieve_memory_context(self, instruction: str, current_file: str = None, errors: str = None) -> str:
        """Retrieve relevant context from memory using CodeContext.

        Args:
            instruction: Task instruction
            current_file: Currently accessed file (if any)
            errors: Current error messages (if any)

        Returns:
            Formatted memory context string
        """
        if not self.memory or not self._enable_memory:
            return ""

        try:
            # Create code-aware context for retrieval
            context = CodeContext(
                user_query=instruction,
                current_file=current_file,
                errors=errors,
                # Could add more context from tmux session if available
                # accessed_files=..., recent_changes=...
            )

            # Query memory system with rich code context
            # CodeMemorySystem uses dependency boosting, recency weighting, and caching
            results = await self.memory.retrieve_relevant_memories(context=context)

            if not results:
                self._logger.debug("No relevant code memories found")
                return ""

            # Use CodeMemorySystem's formatting for clean presentation
            formatted_context = self.memory.format_memories_for_prompt(
                results,
                group_by_file=True,  # Group related code from same file
                include_scores=True   # Show relevance scores
            )

            self._logger.info(f"Retrieved {len(results)} relevant code memories")
            return formatted_context

        except Exception as e:
            self._logger.warning(f"Code memory retrieval failed: {e}", exc_info=True)
            return ""

    async def cleanup_resources(self):
        """Clean up all resources (memory, databases, containers).

        This method should be called after task completion to ensure
        proper cleanup of all resources.
        """
        logger.info("Starting Terminus agent resource cleanup...")

        # Close memory system if it exists
        if self.memory:
            try:
                if hasattr(self.memory, 'close'):
                    self.memory.close()
                    logger.info("Closed memory system")
            except Exception as e:
                logger.warning(f"Failed to close memory system: {e}")

        # Call parent cleanup if available
        if hasattr(super(), 'cleanup_resources'):
            try:
                await super().cleanup_resources()
                logger.info("Called parent cleanup")
            except Exception as e:
                logger.warning(f"Parent cleanup warning: {e}")

        logger.info("Terminus agent resource cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup happens."""
        try:
            # Close memory system if it exists
            if hasattr(self, 'memory') and self.memory:
                if hasattr(self.memory, 'close'):
                    self.memory.close()
        except Exception:
            pass  # Errors in destructor should be silent
