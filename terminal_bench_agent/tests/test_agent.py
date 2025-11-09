"""Tests for Terminal-Bench agent."""

import pytest
import asyncio
from pathlib import Path

# Import agent components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.actions import Action, ActionType, Observation
from agent.executor import Executor
from agent.planner import Planner


class MockTmuxSession:
    """Mock TmuxSession for testing."""

    def __init__(self):
        self.commands = []
        self.output = "test output"

    def send_keys(self, command):
        self.commands.append(command)

    def capture_pane(self):
        return self.output


class TestActions:
    """Test action definitions."""

    def test_action_creation(self):
        """Test creating an action."""
        action = Action(
            action_type=ActionType.BASH_COMMAND,
            command="ls -la",
            reasoning="List files"
        )
        assert action.action_type == ActionType.BASH_COMMAND
        assert action.command == "ls -la"
        assert action.reasoning == "List files"

    def test_action_to_dict(self):
        """Test converting action to dictionary."""
        action = Action(
            action_type=ActionType.READ_FILE,
            command="test.txt"
        )
        action_dict = action.to_dict()
        assert action_dict["action_type"] == "read_file"
        assert action_dict["command"] == "test.txt"

    def test_observation_creation(self):
        """Test creating an observation."""
        action = Action(ActionType.BASH_COMMAND, "pwd")
        obs = Observation(
            action=action,
            output="/home/user",
            success=True
        )
        assert obs.output == "/home/user"
        assert obs.success is True


class TestExecutor:
    """Test executor component."""

    def test_executor_initialization(self):
        """Test initializing executor."""
        session = MockTmuxSession()
        executor = Executor(session)
        assert executor.session == session

    def test_bash_command_execution(self):
        """Test executing bash command."""
        session = MockTmuxSession()
        executor = Executor(session)

        action = Action(ActionType.BASH_COMMAND, "ls -la")
        obs = executor.execute(action)

        assert "ls -la" in session.commands
        assert obs.success is True

    def test_read_file_action(self):
        """Test reading a file."""
        session = MockTmuxSession()
        executor = Executor(session)

        action = Action(ActionType.READ_FILE, "test.txt")
        obs = executor.execute(action)

        # Should have sent a cat command
        assert any("cat" in cmd for cmd in session.commands)


class TestPlanner:
    """Test planner component."""

    @pytest.mark.asyncio
    async def test_planner_initialization(self):
        """Test initializing planner."""
        async def mock_llm(prompt):
            return '{"action_type": "bash", "command": "ls"}'

        planner = Planner(mock_llm)
        assert planner.llm is not None

    @pytest.mark.asyncio
    async def test_create_plan(self):
        """Test creating a plan."""
        async def mock_llm(prompt):
            return '''[
                {"description": "List files", "action_type": "bash",
                 "command": "ls", "expected_outcome": "File list"}
            ]'''

        planner = Planner(mock_llm)
        plan = await planner.create_plan("List all files")

        assert len(plan) > 0
        assert plan[0].action.action_type == ActionType.BASH_COMMAND

    @pytest.mark.asyncio
    async def test_fallback_plan(self):
        """Test fallback plan creation."""
        async def failing_llm(prompt):
            raise Exception("LLM failed")

        planner = Planner(failing_llm)
        plan = await planner.create_plan("Test task")

        # Should create fallback plan
        assert len(plan) > 0


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_plan_and_execute(self):
        """Test creating and executing a plan."""
        async def mock_llm(prompt):
            return '''[
                {"description": "Check dir", "action_type": "bash",
                 "command": "pwd", "expected_outcome": "Directory path"}
            ]'''

        session = MockTmuxSession()
        planner = Planner(mock_llm)
        executor = Executor(session)

        # Create plan
        plan = await planner.create_plan("Check current directory")
        assert len(plan) > 0

        # Execute first step
        action = plan[0].action
        obs = executor.execute(action)

        assert obs.success is True
        assert "pwd" in session.commands


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
