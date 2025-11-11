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
        self.current_plan: List[Step] = []
        self.current_step_index: int = 0
        self.task_description: str = ""
        self.replan_count: int = 0

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
        self.task_description = task_description

        prompt = self._build_planning_prompt(
            task_description,
            environment_info,
            relevant_memories
        )

        try:
            response = await self.llm(prompt)
            steps = self._parse_plan_response(response)
            self.current_plan = steps if steps else self._create_fallback_plan(task_description)
        except Exception as e:
            # Fallback to simple exploration plan
            self.current_plan = self._create_fallback_plan(task_description)

        self.current_step_index = 0
        return self.current_plan

    async def next_action(
        self,
        task_description: str,
        plan: List[Step],
        current_step_index: int,
        recent_observations: List[Observation],
        error: Optional[str] = None
    ) -> Action:
        """Determine next action based on current state.

        Note: This now manages state internally. The plan and current_step_index
        parameters are kept for API compatibility but are not used.

        Args:
            task_description: Original task description
            plan: (Ignored) Current execution plan
            current_step_index: (Ignored) Index of current step
            recent_observations: Recent observations from execution
            error: Optional error from last action

        Returns:
            Next action to take
        """
        # If we have a current step in the plan, use it
        if self.current_step_index < len(self.current_plan):
            current_step = self.current_plan[self.current_step_index]

            # If there was an error, try to adapt (but don't increment step index yet)
            if error:
                return await self._handle_error(
                    self.task_description,
                    current_step,
                    error,
                    recent_observations
                )

            # Increment step index for next call
            self.current_step_index += 1
            return current_step.action

        # Plan exhausted - check if task is actually complete
        if recent_observations:
            # First verify if the task is complete
            is_complete = await self._check_completion(
                self.task_description,
                recent_observations
            )

            if is_complete:
                return Action(
                    action_type=ActionType.DONE,
                    command="",
                    reasoning="Task verified complete"
                )

            # Task not complete - create a new plan
            # Limit replanning to prevent infinite loops
            if self.replan_count >= 3:
                return Action(
                    action_type=ActionType.DONE,
                    command="",
                    reasoning="Maximum replan attempts reached, task incomplete"
                )

            self.replan_count += 1
            await self._replan(recent_observations)

            # Return first step of new plan if available
            if self.current_plan and self.current_step_index < len(self.current_plan):
                current_step = self.current_plan[self.current_step_index]
                self.current_step_index += 1
                return current_step.action

        # Fallback: explore environment
        return Action(
            action_type=ActionType.BASH_COMMAND,
            command="ls -la",
            reasoning="Unable to plan, exploring environment"
        )

    async def _verify_completion(
        self,
        task_description: str,
        recent_observations: List[Observation]
    ) -> Action:
        """Verify if task is complete and decide next action.

        Args:
            task_description: Original task description
            recent_observations: Recent observations from execution

        Returns:
            DONE action if complete, or next action to continue
        """
        prompt = f"""Task: {task_description}

Recent execution history:
{self._format_observations(recent_observations[-10:])}

Analyze whether the task objective has been FULLY COMPLETED.

Consider:
1. Have all required files been created?
2. Does the code work as specified?
3. Have tests been run and passed?
4. Are all requirements from the task met?

If the task is NOT complete, suggest the next action to continue.
If the task IS complete, respond with "DONE".

Respond with JSON:
{{
    "is_complete": true or false,
    "reasoning": "detailed explanation",
    "next_action_type": "bash" or "read_file" or "done",
    "next_command": "command to run (or empty if done)"
}}"""

        try:
            response = await self.llm(prompt)
            data = self._extract_json(response)

            if data.get("is_complete", False):
                return Action(
                    action_type=ActionType.DONE,
                    command="",
                    reasoning=data.get("reasoning", "Task verified complete")
                )

            # Task not complete, continue working
            return Action(
                action_type=ActionType(data.get("next_action_type", "bash")),
                command=data.get("next_command", "ls -la"),
                reasoning=data.get("reasoning", "Continue working on task")
            )
        except Exception as e:
            # If verification fails, continue working
            return Action(
                action_type=ActionType.BASH_COMMAND,
                command="ls -la",
                reasoning="Could not verify completion, continuing task"
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
        prompt = f"""You are an expert coding agent solving Terminal-Bench challenges.

TASK:
{task_description}

ENVIRONMENT: {environment_info or "Linux container with standard development tools"}

INSTRUCTIONS:
1. Read the task carefully and understand ALL requirements
2. Break down the task into concrete, actionable steps
3. Each step should be specific (not vague like "implement solution")
4. Include steps for: exploration, implementation, testing, verification
5. Use specific file paths mentioned in the task requirements
6. Create a COMPLETE plan with 8-15 detailed steps

IMPORTANT:
- Do NOT create generic or vague steps
- Do NOT skip implementation details
- Include specific commands with actual file paths
- Plan should cover the ENTIRE task from start to finish
- Each step should have a clear, measurable outcome
"""

        if relevant_memories:
            prompt += f"\n\nRELEVANT CONTEXT FROM SIMILAR TASKS:\n{relevant_memories}\n"

        prompt += """
Respond with a JSON array. Each step must have:
- description: Clear description of what this step does
- action_type: "bash" (for commands)
- command: The exact command to run with full paths
- expected_outcome: What should result from this step

Example for an R implementation task:
[
    {
        "description": "Check if R is installed and working",
        "action_type": "bash",
        "command": "R --version",
        "expected_outcome": "R version information displayed"
    },
    {
        "description": "Create main R implementation file",
        "action_type": "bash",
        "command": "cat > /app/ars.R << 'EOF'\\n# Adaptive Rejection Sampler\\nars <- function(n, f) {\\n  # Implementation here\\n}\\nEOF",
        "expected_outcome": "ars.R file created with function skeleton"
    },
    ...more implementation steps...,
    {
        "description": "Run tests to verify implementation",
        "action_type": "bash",
        "command": "Rscript -e 'source(\"/app/ars.R\"); test()'",
        "expected_outcome": "All tests pass"
    }
]

Create your DETAILED, COMPLETE plan now (8-15 steps):"""

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
        """Create comprehensive fallback plan when LLM planning fails.

        Args:
            task_description: Task description

        Returns:
            Exploration and setup plan
        """
        return [
            Step(
                description="Check current working directory and list files",
                action=Action(
                    action_type=ActionType.BASH_COMMAND,
                    command="pwd && ls -la",
                    reasoning="Understand current location and available files"
                ),
                expected_outcome="Current directory path and file listing"
            ),
            Step(
                description="Check for task-specific files or instructions",
                action=Action(
                    action_type=ActionType.BASH_COMMAND,
                    command="find . -maxdepth 2 -name '*.md' -o -name 'README*' -o -name 'INSTRUCTIONS*' 2>/dev/null || echo 'No instruction files found'",
                    reasoning="Look for task documentation"
                ),
                expected_outcome="List of instruction files if any"
            ),
            Step(
                description="Check installed development tools",
                action=Action(
                    action_type=ActionType.BASH_COMMAND,
                    command="which python python3 node npm gcc g++ R go java javac || echo 'Checked for dev tools'",
                    reasoning="Identify available programming languages and tools"
                ),
                expected_outcome="Available development tools"
            ),
            Step(
                description="Analyze task requirements",
                action=Action(
                    action_type=ActionType.BASH_COMMAND,
                    command="echo 'Task analysis needed' && ls -la /app 2>/dev/null || echo 'Working directory ready'",
                    reasoning=f"Prepare to work on: {task_description[:100]}..."
                ),
                expected_outcome="Environment ready for task"
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

    async def _check_completion(
        self,
        task_description: str,
        recent_observations: List[Observation]
    ) -> bool:
        """Check if task is complete.

        Args:
            task_description: Original task description
            recent_observations: Recent observations from execution

        Returns:
            True if task is complete, False otherwise
        """
        prompt = f"""Task: {task_description}

Recent execution history:
{self._format_observations(recent_observations[-15:])}

Has the task been FULLY COMPLETED? Consider:
1. Have all required files been created?
2. Does the code work as specified?
3. Have tests been run and passed?
4. Are all requirements from the task met?

Respond ONLY with "YES" if complete or "NO" if not complete."""

        try:
            response = await self.llm(prompt)
            return "YES" in response.upper() and "NO" not in response.upper()
        except Exception:
            return False

    async def _replan(self, recent_observations: List[Observation]) -> None:
        """Create a new plan based on current progress.

        Args:
            recent_observations: Recent observations from execution
        """
        prompt = f"""Task: {self.task_description}

Progress so far:
{self._format_observations(recent_observations[-15:])}

The initial plan has been completed but the task is NOT DONE yet.
Create a NEW plan to continue working towards completing the task.

Focus on what's MISSING:
1. What files still need to be created?
2. What code needs to be written or fixed?
3. What tests need to be run?
4. What requirements are not yet met?

Respond with a JSON array. Each step must have:
- description: Clear description of what this step does
- action_type: "bash" (for commands)
- command: The exact command to run with full paths
- expected_outcome: What should result from this step

Create your DETAILED, SPECIFIC plan now (5-10 steps):"""

        try:
            response = await self.llm(prompt)
            steps = self._parse_plan_response(response)
            if steps:
                self.current_plan = steps
                self.current_step_index = 0
            else:
                # Fallback to simple exploration
                self.current_plan = [
                    Step(
                        description="Continue working on task",
                        action=Action(
                            action_type=ActionType.BASH_COMMAND,
                            command="pwd && ls -la",
                            reasoning="Reassess current state"
                        ),
                        expected_outcome="Current directory listing"
                    )
                ]
                self.current_step_index = 0
        except Exception:
            # Fallback to simple exploration
            self.current_plan = [
                Step(
                    description="Continue working on task",
                    action=Action(
                        action_type=ActionType.BASH_COMMAND,
                        command="pwd && ls -la",
                        reasoning="Reassess current state"
                    ),
                    expected_outcome="Current directory listing"
                )
            ]
            self.current_step_index = 0
