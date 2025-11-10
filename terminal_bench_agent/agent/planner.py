"""Planner for creating and adapting execution plans."""

import json
from typing import List, Optional, Dict, Any
from .actions import Action, ActionType, Step, Observation


class Planner:
    """Creates and updates execution plans using LLM."""

    def __init__(self, llm_function, memory_system=None):
        """Initialize planner.

        Args:
            llm_function: Async function to call LLM
            memory_system: Optional memory system for retrieving context
        """
        self.llm = llm_function
        self.memory = memory_system

    async def create_plan(
        self,
        task_description: str,
        environment_info: Optional[str] = None,
        relevant_memories: Optional[str] = None
    ) -> List[Step]:
        """Create initial plan for task.

        Args:
            task_description: Description of the task
            environment_info: Information about environment
            relevant_memories: Relevant memories from memory system

        Returns:
            List of steps to execute
        """
        prompt = self._build_planning_prompt(
            task_description,
            environment_info,
            relevant_memories
        )

        try:
            response = await self.llm(prompt)
            steps = self._parse_plan_response(response)
            return steps
        except Exception as e:
            # Fallback to simple exploration plan
            return self._create_fallback_plan(task_description)

    async def next_action(
        self,
        task_description: str,
        plan: List[Step],
        current_step_index: int,
        recent_observations: List[Observation],
        error: Optional[str] = None
    ) -> Action:
        """Determine next action based on current state.

        Args:
            task_description: Original task description
            plan: Current execution plan
            current_step_index: Index of current step
            recent_observations: Recent observations from execution
            error: Optional error from last action

        Returns:
            Next action to take
        """
        # If we have a current step in the plan, use it
        if current_step_index < len(plan):
            current_step = plan[current_step_index]

            # If there was an error, try to adapt
            if error:
                return await self._handle_error(
                    task_description,
                    current_step,
                    error,
                    recent_observations
                )

            return current_step.action

        # No more steps in plan, create completion action
        return Action(
            action_type=ActionType.DONE,
            command="",
            reasoning="Plan completed"
        )

    async def _handle_error(
        self,
        task_description: str,
        failed_step: Step,
        error: str,
        recent_observations: List[Observation]
    ) -> Action:
        """Handle error by creating recovery action.

        Args:
            task_description: Original task
            failed_step: Step that failed
            error: Error message
            recent_observations: Recent observations

        Returns:
            Recovery action
        """
        prompt = f"""The following step failed:
Step: {failed_step.description}
Action: {failed_step.action.command}
Error: {error}

Task: {task_description}

Recent context:
{self._format_observations(recent_observations[-3:])}

Suggest a recovery action. Respond with JSON:
{{
    "action_type": "bash" or "read_file" or "list_dir",
    "command": "the command to run",
    "reasoning": "why this will help"
}}"""

        try:
            response = await self.llm(prompt)
            action_data = self._extract_json(response)

            return Action(
                action_type=ActionType(action_data.get("action_type", "bash")),
                command=action_data.get("command", "pwd"),
                reasoning=action_data.get("reasoning", "Recovery action")
            )
        except Exception:
            # Fallback: list current directory to understand state
            return Action(
                action_type=ActionType.LIST_DIR,
                command=".",
                reasoning="Error occurred, listing directory to understand state"
            )

    def _build_planning_prompt(
        self,
        task_description: str,
        environment_info: Optional[str],
        relevant_memories: Optional[str]
    ) -> str:
        """Build prompt for plan generation.

        Args:
            task_description: Task description
            environment_info: Environment information
            relevant_memories: Relevant memories

        Returns:
            Planning prompt
        """
        prompt = f"""You are a Terminal-Bench coding agent. Create a step-by-step plan.

Task: {task_description}"""

        if environment_info:
            prompt += f"\n\nEnvironment: {environment_info}"

        if relevant_memories:
            prompt += f"\n\nRelevant past solutions:\n{relevant_memories}"

        prompt += """

Create a detailed plan with 3-10 steps. For each step, specify:
1. Description: What to do
2. Action type: bash, read_file, list_dir, etc.
3. Command: The actual command
4. Expected outcome: What should happen

Respond with a JSON array of steps:
[
    {
        "description": "Check current directory",
        "action_type": "bash",
        "command": "pwd && ls -la",
        "expected_outcome": "See current directory contents"
    },
    ...
]

Plan:"""

        return prompt

    def _parse_plan_response(self, response: str) -> List[Step]:
        """Parse LLM response into steps.

        Args:
            response: LLM response

        Returns:
            List of steps
        """
        try:
            # Extract JSON from response
            steps_data = self._extract_json(response)

            if isinstance(steps_data, dict):
                steps_data = [steps_data]

            steps = []
            for step_data in steps_data:
                action = Action(
                    action_type=ActionType(step_data.get("action_type", "bash")),
                    command=step_data.get("command", ""),
                    reasoning=step_data.get("description", "")
                )

                step = Step(
                    description=step_data.get("description", ""),
                    action=action,
                    expected_outcome=step_data.get("expected_outcome", ""),
                    verification=step_data.get("verification", "")
                )
                steps.append(step)

            return steps

        except Exception as e:
            # Parse failed, create fallback plan
            return []

    def _create_fallback_plan(self, task_description: str) -> List[Step]:
        """Create simple fallback plan.

        Args:
            task_description: Task description

        Returns:
            Simple exploration plan
        """
        return [
            Step(
                description="Check current directory",
                action=Action(
                    action_type=ActionType.BASH_COMMAND,
                    command="pwd",
                    reasoning="Understand current location"
                ),
                expected_outcome="Current directory path"
            ),
            Step(
                description="List files",
                action=Action(
                    action_type=ActionType.LIST_DIR,
                    command=".",
                    reasoning="See available files"
                ),
                expected_outcome="Directory listing"
            ),
            Step(
                description="Explore task",
                action=Action(
                    action_type=ActionType.THINK,
                    command="",
                    reasoning=f"Analyze task: {task_description}"
                ),
                expected_outcome="Understanding of task"
            )
        ]

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from text.

        Args:
            text: Text potentially containing JSON

        Returns:
            Parsed JSON object
        """
        # Try to find JSON array or object
        import re

        # Look for JSON array
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            return json.loads(array_match.group(0))

        # Look for JSON object
        obj_match = re.search(r'\{[\s\S]*\}', text)
        if obj_match:
            return json.loads(obj_match.group(0))

        # Try parsing entire text
        return json.loads(text)

    def _format_observations(self, observations: List[Observation]) -> str:
        """Format observations for inclusion in prompt.

        Args:
            observations: List of observations

        Returns:
            Formatted string
        """
        formatted = []
        for obs in observations:
            formatted.append(f"Action: {obs.action.command}")
            formatted.append(f"Output: {obs.output[:200]}")
            if obs.error:
                formatted.append(f"Error: {obs.error}")
            formatted.append("")

        return "\n".join(formatted)
