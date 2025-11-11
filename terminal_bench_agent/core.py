"""Core agent with memory integration for Harbor/Terminal-Bench 2.0.

This module implements Harbor's BaseAgent interface.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

# Harbor imports
try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
    HARBOR_AVAILABLE = True
except ImportError:
    HARBOR_AVAILABLE = False
    BaseAgent = object

from .agent.planner import Planner
from .agent.executor import Executor
from .agent.actions import ActionType, Observation

# OpenAI client for LLM calls
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class MemoryGuidedAgent(BaseAgent if HARBOR_AVAILABLE else object):
    """Memory-guided agent for Terminal-Bench tasks via Harbor.

    This agent uses a plan-execute-observe loop to solve tasks,
    with optional memory system integration for code intelligence.
    """

    @staticmethod
    def name() -> str:
        """Return agent name for Harbor."""
        return "memory-guided-agent"

    def version(self) -> str | None:
        """Return agent version."""
        return "0.1.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: Optional[str] = None,
        max_steps: int = 50,
        max_time_seconds: int = 600,
        *args,
        **kwargs
    ):
        """Initialize the agent.

        Args:
            logs_dir: Directory for agent logs
            model_name: Name of the LLM model to use (e.g., "gpt-5-codex")
            max_steps: Maximum number of execution steps
            max_time_seconds: Maximum time for task execution
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        if HARBOR_AVAILABLE:
            super().__init__(logs_dir, model_name, *args, **kwargs)

        # Store configuration
        self.logs_dir = logs_dir
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4")
        self.max_steps = max_steps
        self.max_time_seconds = max_time_seconds

        # Initialize OpenAI client
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required. Install with: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = AsyncOpenAI(api_key=api_key)

        # Create LLM function
        self.llm = self._create_llm_function()

        # Memory system (not initialized - could be added later)
        self.memory = None

        # Will be initialized per task
        self.planner = None
        self.executor = None
        self.environment = None

    def _create_llm_function(self) -> Callable:
        """Create an async LLM function for the agent.

        Returns:
            Async function that calls the LLM
        """
        async def llm_function(prompt: str, **kwargs) -> str:
            """Call the LLM with a prompt.

            Args:
                prompt: The prompt to send to the LLM
                **kwargs: Additional arguments for the API call

            Returns:
                The LLM's response text
            """
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 4000)
                )
                return response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"LLM call failed: {e}")

        return llm_function

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup the agent in the environment.

        Args:
            environment: The Harbor environment
        """
        self.environment = environment
        # No special setup needed for this agent
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run the agent to solve a task.

        Args:
            instruction: The task instruction
            environment: The Harbor environment
            context: The agent context to populate with results
        """
        self.environment = environment

        # Initialize components for this task
        # Note: For now, we'll execute commands directly in the environment
        # In a full implementation, we'd need a proper Executor that works with Harbor's environment

        start_time = datetime.now()
        observations = []
        current_step = 0

        try:
            # Get environment info
            env_info = "Docker container with standard Linux environment"

            # Create initial plan
            self.planner = Planner(self.llm, self.memory)
            plan = await self.planner.create_plan(
                instruction,
                env_info,
                None  # No memory retrieval for now
            )

            # Log initial state
            self._log(f"Task: {instruction}\n")
            self._log(f"Plan: {len(plan)} steps\n\n")

            # Main execution loop
            while current_step < self.max_steps:
                # Check timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > self.max_time_seconds:
                    self._log(f"\nTimeout after {elapsed}s\n")
                    break

                # Get next action
                last_error = None
                if observations and not observations[-1].success:
                    last_error = observations[-1].error

                action = await self.planner.next_action(
                    instruction,
                    plan,
                    current_step,
                    observations[-5:] if observations else [],
                    last_error
                )

                # Check if done
                if action.action_type == ActionType.DONE:
                    self._log("\nAgent marked task as DONE\n")
                    break

                # Execute action in environment
                if action.action_type == ActionType.BASH_COMMAND and action.command:
                    # Execute bash command in Harbor environment
                    try:
                        result = await environment.exec(action.command)
                        success = result.return_code == 0
                        output_text = result.stdout or ""
                        error_text = result.stderr if result.return_code != 0 else None

                        observation = Observation(
                            action=action,
                            output=output_text,
                            success=success,
                            error=error_text
                        )
                    except Exception as e:
                        observation = Observation(
                            action=action,
                            output="",
                            success=False,
                            error=str(e)
                        )

                    observations.append(observation)

                    # Log execution
                    self._log(f"\nStep {current_step + 1}:\n")
                    self._log(f"Command: {action.command}\n")
                    self._log(f"Return Code: {result.return_code if 'result' in locals() else 'N/A'}\n")
                    self._log(f"Success: {observation.success}\n")
                    if observation.output:
                        self._log(f"Output: {observation.output[:500]}\n")
                    if observation.error:
                        self._log(f"Error: {observation.error}\n")

                current_step += 1

                # Small delay
                await asyncio.sleep(0.1)

            # Return result via context
            total_time = (datetime.now() - start_time).total_seconds()
            self._log(f"\n\nCompleted in {total_time}s with {current_step} steps\n")

            # Populate context with execution info
            # Harbor will run verifiers to determine actual success

        except Exception as e:
            self._log(f"\nError during execution: {e}\n")
            raise

    def _log(self, message: str):
        """Write to log file.

        Args:
            message: Message to log
        """
        try:
            log_file = self.logs_dir / "agent_execution.log"
            with open(log_file, "a") as f:
                f.write(message)
        except Exception:
            pass


# For standalone testing
async def test_agent_standalone():
    """Test agent in standalone mode."""
    print("Testing Harbor agent...")

    from pathlib import Path

    agent = MemoryGuidedAgent(
        logs_dir=Path("/tmp/harbor_test"),
        model_name="gpt-4",
        max_steps=10
    )

    print(f"Agent: {agent.name()}")
    print(f"Version: {agent.version()}")
    print(f"Model: {agent.model_name}")
    print("\nAgent initialized successfully!")


if __name__ == "__main__":
    asyncio.run(test_agent_standalone())
