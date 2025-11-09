"""Core Terminal-Bench agent with memory integration.

This module implements the Terminal-Bench BaseAgent interface.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# Terminal-Bench imports (will be available when installed)
try:
    from terminal_bench.agents import BaseAgent, AgentResult
    from terminal_bench.terminal.tmux_session import TmuxSession
    TBENCH_AVAILABLE = True
except ImportError:
    # For development without Terminal-Bench installed
    TBENCH_AVAILABLE = False
    BaseAgent = object
    AgentResult = dict
    TmuxSession = object

from .agent.planner import Planner
from .agent.executor import Executor
from .agent.actions import ActionType, Observation


class MemoryGuidedAgent(BaseAgent if TBENCH_AVAILABLE else object):
    """Terminal-Bench agent with memory-guided execution.

    This agent uses the memory system for code intelligence and implements
    a plan-execute-observe loop to solve tasks.
    """

    @staticmethod
    def name() -> str:
        """Return agent name for Terminal-Bench."""
        return "memory-guided-agent"

    def __init__(
        self,
        llm_function,
        memory_system=None,
        max_steps: int = 50,
        max_time_seconds: int = 600,
        **kwargs
    ):
        """Initialize the agent.

        Args:
            llm_function: Async function to call LLM for planning/execution
            memory_system: Optional CodeMemorySystem for memory integration
            max_steps: Maximum number of execution steps
            max_time_seconds: Maximum time for task execution
            **kwargs: Additional arguments
        """
        if TBENCH_AVAILABLE:
            super().__init__(**kwargs)

        self.llm = llm_function
        self.memory = memory_system
        self.max_steps = max_steps
        self.max_time_seconds = max_time_seconds

        # Will be initialized per task
        self.planner = None
        self.executor = None

    def perform_task(
        self,
        task_description: str,
        session: TmuxSession,
        logging_dir: Optional[Path] = None,
    ) -> AgentResult:
        """Perform a Terminal-Bench task.

        This is the main entry point called by Terminal-Bench harness.

        Args:
            task_description: Natural language description of the task
            session: TmuxSession for interacting with terminal
            logging_dir: Optional directory for logging

        Returns:
            AgentResult with success status and metadata
        """
        # Initialize components for this task
        self.planner = Planner(self.llm, self.memory)
        self.executor = Executor(session)

        # Run async task in event loop
        try:
            result = asyncio.run(self._solve_task_async(
                task_description,
                logging_dir
            ))
            return result
        except Exception as e:
            return AgentResult(
                success=False,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            ) if TBENCH_AVAILABLE else {
                "success": False,
                "error": str(e)
            }

    async def _solve_task_async(
        self,
        task_description: str,
        logging_dir: Optional[Path]
    ) -> AgentResult:
        """Async task solving logic.

        Args:
            task_description: Task to solve
            logging_dir: Logging directory

        Returns:
            AgentResult
        """
        start_time = datetime.now()
        observations = []
        current_step = 0

        # Get environment info
        env_info = self._get_environment_info()

        # Retrieve relevant memories if memory system available
        relevant_memories = None
        if self.memory:
            relevant_memories = await self._retrieve_memories(task_description)

        # Create initial plan
        plan = await self.planner.create_plan(
            task_description,
            env_info,
            relevant_memories
        )

        # Log initial state
        if logging_dir:
            self._log(logging_dir, f"Task: {task_description}\n")
            self._log(logging_dir, f"Plan: {len(plan)} steps\n\n")

        # Main execution loop
        while current_step < self.max_steps:
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.max_time_seconds:
                if logging_dir:
                    self._log(logging_dir, f"\nTimeout after {elapsed}s\n")
                break

            # Get next action
            last_error = None
            if observations and not observations[-1].success:
                last_error = observations[-1].error

            action = await self.planner.next_action(
                task_description,
                plan,
                current_step,
                observations[-5:],  # Last 5 observations
                last_error
            )

            # Check if done
            if action.action_type == ActionType.DONE:
                if logging_dir:
                    self._log(logging_dir, "\nAgent marked task as DONE\n")
                break

            # Execute action
            observation = self.executor.execute(action)
            observations.append(observation)

            # Log execution
            if logging_dir:
                self._log(logging_dir, f"\nStep {current_step + 1}:\n")
                self._log(logging_dir, f"Action: {action.action_type.value}\n")
                self._log(logging_dir, f"Command: {action.command}\n")
                self._log(logging_dir, f"Success: {observation.success}\n")
                if observation.output:
                    self._log(logging_dir, f"Output: {observation.output[:500]}\n")
                if observation.error:
                    self._log(logging_dir, f"Error: {observation.error}\n")

            current_step += 1

            # Small delay to avoid overwhelming the terminal
            await asyncio.sleep(0.1)

        # Store learnings in memory if successful
        if self.memory and observations:
            await self._store_learnings(
                task_description,
                observations,
                success=True  # Will be determined by Terminal-Bench tests
            )

        # Return result
        total_time = (datetime.now() - start_time).total_seconds()

        result_metadata = {
            "total_steps": current_step,
            "total_time_seconds": total_time,
            "observations_count": len(observations),
            "plan_steps": len(plan)
        }

        if logging_dir:
            self._log(logging_dir, f"\n\nCompleted in {total_time}s with {current_step} steps\n")

        # Terminal-Bench will run tests to determine actual success
        # We return our best attempt
        return AgentResult(
            success=True,  # Tests will determine real success
            metadata=result_metadata
        ) if TBENCH_AVAILABLE else result_metadata

    async def _retrieve_memories(self, task_description: str) -> Optional[str]:
        """Retrieve relevant memories for task.

        Args:
            task_description: Task description

        Returns:
            Formatted memories or None
        """
        if not self.memory:
            return None

        try:
            from memory_lib.codebase import CodeContext

            context = CodeContext(
                user_query=task_description,
                additional_context="Terminal-Bench task"
            )

            memories = await self.memory.retrieve_relevant_memories(context)

            if memories:
                return self.memory.format_memories_for_prompt(
                    memories,
                    include_scores=True
                )

        except Exception as e:
            # Memory retrieval failed, continue without
            pass

        return None

    async def _store_learnings(
        self,
        task_description: str,
        observations: list,
        success: bool
    ):
        """Store task solution in memory.

        Args:
            task_description: Task that was solved
            observations: Observations from execution
            success: Whether task was successful
        """
        if not self.memory:
            return

        try:
            # Extract commands used
            commands = [
                obs.action.command
                for obs in observations
                if obs.action.command and obs.action.action_type != ActionType.THINK
            ]

            # Create solution summary
            solution_content = f"""Task: {task_description}

Success: {success}

Steps taken: {len(observations)}

Commands used:
{chr(10).join(f"- {cmd}" for cmd in commands[:10])}

Key insights:
{'This solution worked.' if success else 'This approach did not work.'}
"""

            # Store as documentation memory
            self.memory.add_documentation_memory(
                title=f"Terminal-Bench Task: {task_description[:50]}",
                content=solution_content,
                category="terminal_bench_task",
                metadata={
                    "success": success,
                    "steps": len(observations),
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            # Failed to store, but don't fail the task
            pass

    def _get_environment_info(self) -> str:
        """Get basic environment information.

        Returns:
            Environment description
        """
        return "Docker container with standard Linux environment"

    def _log(self, logging_dir: Path, message: str):
        """Write to log file.

        Args:
            logging_dir: Directory for logs
            message: Message to log
        """
        if not logging_dir:
            return

        try:
            log_file = logging_dir / "agent_execution.log"
            with open(log_file, "a") as f:
                f.write(message)
        except Exception:
            pass


# For standalone testing without Terminal-Bench
async def test_agent_standalone():
    """Test agent without Terminal-Bench harness."""
    print("Testing agent in standalone mode...")

    async def mock_llm(prompt: str) -> str:
        """Mock LLM for testing."""
        # Simple plan generation
        if "Create a detailed plan" in prompt:
            return """[
                {
                    "description": "Check current directory",
                    "action_type": "bash",
                    "command": "pwd",
                    "expected_outcome": "Current directory path"
                },
                {
                    "description": "List files",
                    "action_type": "bash",
                    "command": "ls -la",
                    "expected_outcome": "Directory contents"
                },
                {
                    "description": "Complete task",
                    "action_type": "done",
                    "command": "",
                    "expected_outcome": "Task finished"
                }
            ]"""
        return '{"action_type": "bash", "command": "ls", "reasoning": "Explore"}'

    class MockSession:
        """Mock TmuxSession for testing."""
        def send_keys(self, command):
            print(f"  [EXEC] {command}")

        def capture_pane(self):
            return "/home/user\ntotal 8\ndrwxr-xr-x 2 user user 4096 Jan 1 12:00 .\ndrwxr-xr-x 3 user user 4096 Jan 1 12:00 .."

    agent = MemoryGuidedAgent(
        llm_function=mock_llm,
        memory_system=None,
        max_steps=10
    )

    print("\nRunning test task...")
    result = agent.perform_task(
        task_description="List all files in the current directory",
        session=MockSession(),
        logging_dir=Path("/tmp/agent_test")
    )

    print(f"\nResult: {result}")
    print("\nTest completed!")


if __name__ == "__main__":
    # Run standalone test
    asyncio.run(test_agent_standalone())
