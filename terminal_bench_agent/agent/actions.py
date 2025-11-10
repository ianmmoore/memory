"""Action definitions for the Terminal-Bench agent."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class ActionType(Enum):
    """Types of actions the agent can perform."""

    # Terminal commands
    BASH_COMMAND = "bash"

    # File operations
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EDIT_FILE = "edit_file"

    # Directory operations
    LIST_DIR = "list_dir"
    CHANGE_DIR = "change_dir"

    # Code operations
    RUN_PYTHON = "run_python"
    RUN_SCRIPT = "run_script"

    # Information gathering
    SEARCH_FILES = "search_files"
    INSPECT_FILE = "inspect_file"

    # Reasoning
    THINK = "think"
    PLAN = "plan"

    # Control
    DONE = "done"


@dataclass
class Action:
    """Represents a single action to be performed."""

    action_type: ActionType
    command: str
    reasoning: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "action_type": self.action_type.value,
            "command": self.command,
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create action from dictionary."""
        return cls(
            action_type=ActionType(data["action_type"]),
            command=data["command"],
            reasoning=data.get("reasoning", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class Observation:
    """Observation from executing an action."""

    action: Action
    output: str
    error: Optional[str] = None
    exit_code: int = 0
    success: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary."""
        return {
            "action": self.action.to_dict(),
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
            "success": self.success,
            "metadata": self.metadata
        }


@dataclass
class Step:
    """A step in the execution plan."""

    description: str
    action: Action
    expected_outcome: str = ""
    verification: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "description": self.description,
            "action": self.action.to_dict(),
            "expected_outcome": self.expected_outcome,
            "verification": self.verification
        }
