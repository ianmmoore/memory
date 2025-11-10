"""Executor for running actions in Terminal-Bench environment."""

import re
import time
from pathlib import Path
from typing import Optional, Tuple
from .actions import Action, ActionType, Observation


class Executor:
    """Executes actions in the terminal environment."""

    def __init__(self, session, max_command_length: int = 10000):
        """Initialize executor.

        Args:
            session: TmuxSession from Terminal-Bench
            max_command_length: Maximum length for commands
        """
        self.session = session
        self.max_command_length = max_command_length
        self.working_directory = "/"

    def execute(self, action: Action) -> Observation:
        """Execute an action and return observation.

        Args:
            action: Action to execute

        Returns:
            Observation with results
        """
        if action.action_type == ActionType.BASH_COMMAND:
            return self._execute_bash(action)
        elif action.action_type == ActionType.READ_FILE:
            return self._read_file(action)
        elif action.action_type == ActionType.WRITE_FILE:
            return self._write_file(action)
        elif action.action_type == ActionType.LIST_DIR:
            return self._list_dir(action)
        elif action.action_type == ActionType.THINK:
            return self._think(action)
        elif action.action_type == ActionType.DONE:
            return self._done(action)
        else:
            # Default to bash command
            return self._execute_bash(action)

    def _execute_bash(self, action: Action) -> Observation:
        """Execute a bash command.

        Args:
            action: Action with bash command

        Returns:
            Observation with command output
        """
        command = action.command.strip()

        # Safety check
        if len(command) > self.max_command_length:
            return Observation(
                action=action,
                output="",
                error=f"Command too long ({len(command)} > {self.max_command_length})",
                exit_code=1,
                success=False
            )

        # Track directory changes
        if command.startswith("cd "):
            self.working_directory = command[3:].strip()

        try:
            # Send command to tmux session
            self.session.send_keys(command)

            # Wait for output
            time.sleep(0.5)

            # Get output
            output = self.session.capture_pane()

            # Simple success detection (can be improved)
            success = not self._is_error_output(output)
            exit_code = 0 if success else 1

            return Observation(
                action=action,
                output=output,
                error=None if success else "Command may have failed",
                exit_code=exit_code,
                success=success
            )

        except Exception as e:
            return Observation(
                action=action,
                output="",
                error=str(e),
                exit_code=1,
                success=False
            )

    def _read_file(self, action: Action) -> Observation:
        """Read a file using cat.

        Args:
            action: Action with file path in command

        Returns:
            Observation with file contents
        """
        file_path = action.command
        bash_action = Action(
            action_type=ActionType.BASH_COMMAND,
            command=f"cat {file_path}",
            reasoning=f"Reading file: {file_path}"
        )
        return self._execute_bash(bash_action)

    def _write_file(self, action: Action) -> Observation:
        """Write content to a file.

        Args:
            action: Action with metadata containing 'file_path' and 'content'

        Returns:
            Observation with write result
        """
        file_path = action.metadata.get("file_path", action.command)
        content = action.metadata.get("content", "")

        # Escape content for heredoc
        escaped_content = content.replace("'", "'\"'\"'")

        bash_action = Action(
            action_type=ActionType.BASH_COMMAND,
            command=f"cat > {file_path} << 'EOF'\n{escaped_content}\nEOF",
            reasoning=f"Writing to file: {file_path}"
        )
        return self._execute_bash(bash_action)

    def _list_dir(self, action: Action) -> Observation:
        """List directory contents.

        Args:
            action: Action with directory path (or current dir if empty)

        Returns:
            Observation with directory listing
        """
        dir_path = action.command if action.command else "."
        bash_action = Action(
            action_type=ActionType.BASH_COMMAND,
            command=f"ls -la {dir_path}",
            reasoning=f"Listing directory: {dir_path}"
        )
        return self._execute_bash(bash_action)

    def _think(self, action: Action) -> Observation:
        """Thinking/reasoning step (no execution).

        Args:
            action: Action with reasoning

        Returns:
            Observation indicating thinking step
        """
        return Observation(
            action=action,
            output=f"Reasoning: {action.reasoning}",
            error=None,
            exit_code=0,
            success=True,
            metadata={"is_thinking": True}
        )

    def _done(self, action: Action) -> Observation:
        """Mark task as complete.

        Args:
            action: Done action

        Returns:
            Observation indicating completion
        """
        return Observation(
            action=action,
            output="Task completed",
            error=None,
            exit_code=0,
            success=True,
            metadata={"is_done": True}
        )

    def _is_error_output(self, output: str) -> bool:
        """Check if output indicates an error.

        Args:
            output: Command output

        Returns:
            True if output appears to be an error
        """
        error_patterns = [
            r"error:",
            r"Error:",
            r"ERROR:",
            r"command not found",
            r"No such file or directory",
            r"Permission denied",
            r"cannot",
            r"failed",
            r"Traceback"
        ]

        for pattern in error_patterns:
            if re.search(pattern, output):
                return True

        return False

    def get_current_directory(self) -> str:
        """Get current working directory.

        Returns:
            Current working directory path
        """
        # Execute pwd to get actual directory
        action = Action(
            action_type=ActionType.BASH_COMMAND,
            command="pwd",
            reasoning="Get current directory"
        )
        obs = self._execute_bash(action)
        if obs.success:
            return obs.output.strip()
        return self.working_directory
