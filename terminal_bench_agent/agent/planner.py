"""Planner for creating and adapting execution plans."""

import json
import logging
from typing import List, Optional, Dict, Any
from .actions import Action, ActionType, Step, Observation

# Set up logger
logger = logging.getLogger(__name__)


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
            if steps:
                self.current_plan = steps
                logger.info(f"Created plan with {len(steps)} steps")
            else:
                logger.warning("LLM returned empty plan, using fallback exploration plan")
                self.current_plan = self._create_fallback_plan(task_description)
        except Exception as e:
            # Fallback to simple exploration plan
            logger.error(f"LLM planning failed: {type(e).__name__}: {e}")
            logger.warning("Falling back to exploration plan")
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
            # Limit replanning to prevent infinite loops (increased from 3 to 10)
            if self.replan_count >= 10:
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

YOUR OBJECTIVE:
Create a DETAILED, COMPLETE implementation plan with 8-15 specific steps. This plan must include:
- Environment exploration (2-3 steps): Check versions, inspect files, understand structure
- Implementation (5-8 steps): Actual coding, file creation, configuration
- Testing & verification (2-3 steps): Run tests, verify outputs, check requirements

CRITICAL REQUIREMENTS:
1. Provide EXACTLY 8-15 steps (NOT fewer!)
2. Each step must be SPECIFIC with exact commands and file paths
3. NO vague steps like "implement solution" or "fix issues"
4. Include FULL implementation details, not just exploratory commands
5. Use exact paths from the task description
"""

        if relevant_memories:
            prompt += f"\n\nRELEVANT CONTEXT FROM SIMILAR TASKS:\n{relevant_memories}\n"

        prompt += """
RESPONSE FORMAT - START YOUR RESPONSE WITH THE JSON ARRAY IMMEDIATELY:

JSON Schema (required fields):
[
    {
        "description": "string - what this step accomplishes",
        "action_type": "bash",
        "command": "string - exact bash command with full paths",
        "expected_outcome": "string - what should happen"
    }
]

Example comprehensive plan (10 steps):
[
    {
        "description": "Check if Python is installed and get version",
        "action_type": "bash",
        "command": "python3 --version && which python3",
        "expected_outcome": "Python version 3.x displayed with path"
    },
    {
        "description": "Check current directory and list contents",
        "action_type": "bash",
        "command": "pwd && ls -la",
        "expected_outcome": "Working directory and file listing shown"
    },
    {
        "description": "Inspect task requirements file if present",
        "action_type": "bash",
        "command": "cat README.md || cat instructions.txt || ls -R",
        "expected_outcome": "Task requirements displayed or directory structure shown"
    },
    {
        "description": "Create main implementation file with basic structure",
        "action_type": "bash",
        "command": "cat > /app/solution.py << 'EOF'\\nimport sys\\n\\ndef main():\\n    # TODO: Implementation\\n    pass\\n\\nif __name__ == '__main__':\\n    main()\\nEOF",
        "expected_outcome": "solution.py created with skeleton code"
    },
    {
        "description": "Implement core algorithm logic",
        "action_type": "bash",
        "command": "cat >> /app/solution.py << 'EOF'\\n# Core logic implementation\\nEOF",
        "expected_outcome": "Core logic added to solution.py"
    },
    {
        "description": "Add input/output handling",
        "action_type": "bash",
        "command": "sed -i 's/pass/implementation_code()/' /app/solution.py",
        "expected_outcome": "I/O handling integrated"
    },
    {
        "description": "Install required dependencies if needed",
        "action_type": "bash",
        "command": "pip install --no-cache-dir -r requirements.txt 2>/dev/null || echo 'No requirements file'",
        "expected_outcome": "Dependencies installed or skipped"
    },
    {
        "description": "Run basic syntax check",
        "action_type": "bash",
        "command": "python3 -m py_compile /app/solution.py",
        "expected_outcome": "No syntax errors reported"
    },
    {
        "description": "Execute tests if test file exists",
        "action_type": "bash",
        "command": "python3 -m pytest tests/ || python3 test.py || echo 'No tests found'",
        "expected_outcome": "Tests run and results displayed"
    },
    {
        "description": "Verify final solution works with sample input",
        "action_type": "bash",
        "command": "echo 'sample input' | python3 /app/solution.py",
        "expected_outcome": "Correct output produced"
    }
]

NOW CREATE YOUR 8-15 STEP PLAN. Begin your response with '[' - NO explanatory text before the JSON:"""

        return prompt

    def _parse_plan_response(self, response: str) -> List[Step]:
        """Parse LLM response into steps.

        Args:
            response: LLM response

        Returns:
            List of steps (calls fallback if parsing fails)
        """
        # Log full response for debugging (not just 500 chars)
        logger.debug(f"Parsing LLM response ({len(response)} chars)")
        logger.debug(f"Full LLM response:\n{response}")

        try:
            # Extract JSON from response
            steps_data = self._extract_json(response)

            if isinstance(steps_data, dict):
                logger.info("LLM returned single step as dict, converting to list")
                steps_data = [steps_data]

            if not isinstance(steps_data, list):
                raise ValueError(f"Expected list or dict, got {type(steps_data)}")

            # Validate minimum step count
            if len(steps_data) < 3:
                logger.warning(f"Plan has only {len(steps_data)} steps (expected ≥3), may be incomplete")

            steps = []
            for i, step_data in enumerate(steps_data):
                # Validate required fields
                if not isinstance(step_data, dict):
                    logger.warning(f"Step {i} is not a dict, skipping: {step_data}")
                    continue

                description = step_data.get("description", "")
                command = step_data.get("command", "")

                if not description or not command:
                    logger.warning(f"Step {i} missing required fields (description or command), skipping")
                    continue

                # Handle ActionType enum safely
                action_type_str = step_data.get("action_type", "bash")
                try:
                    # Try direct enum lookup by value
                    action_type = ActionType(action_type_str)
                except ValueError:
                    # Fallback: try by name
                    logger.warning(f"Invalid action_type '{action_type_str}', defaulting to BASH_COMMAND")
                    action_type = ActionType.BASH_COMMAND

                action = Action(
                    action_type=action_type,
                    command=command,
                    reasoning=description
                )

                step = Step(
                    description=description,
                    action=action,
                    expected_outcome=step_data.get("expected_outcome", ""),
                    verification=step_data.get("verification", "")
                )
                steps.append(step)

            if not steps:
                raise ValueError("No valid steps extracted from LLM response")

            logger.info(f"Successfully parsed {len(steps)} valid steps from LLM response")
            return steps

        except Exception as e:
            # Parse failed, create fallback plan
            logger.error(f"Failed to parse LLM plan response: {type(e).__name__}: {e}")
            logger.error(f"Response preview: {response[:300]}...")
            logger.warning("Creating fallback exploration plan")
            return self._create_fallback_plan(self.task_description)

    def _create_fallback_plan(self, task_description: str) -> List[Step]:
        """Create comprehensive fallback plan when LLM planning fails.

        This plan should NOT just explore - it should actually attempt basic implementation.

        Args:
            task_description: Task description

        Returns:
            Exploration and setup plan with implementation steps
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
                    command="find . -maxdepth 3 -type f '(' -name '*.md' -o -name 'README*' -o -name 'INSTRUCTIONS*' -o -name 'requirements.txt' -o -name 'package.json' ')' 2>/dev/null | head -20",
                    reasoning="Look for task documentation and dependencies"
                ),
                expected_outcome="List of instruction and config files if any"
            ),
            Step(
                description="Check installed development tools and versions",
                action=Action(
                    action_type=ActionType.BASH_COMMAND,
                    command="echo '=== Languages ===' && python3 --version 2>&1 && node --version 2>&1 && R --version 2>&1 | head -1 && go version 2>&1 && rustc --version 2>&1 && gcc --version 2>&1 | head -1 || echo 'Tool check complete'",
                    reasoning="Identify available programming languages and versions"
                ),
                expected_outcome="Available development tools with versions"
            ),
            Step(
                description="Check /app directory structure",
                action=Action(
                    action_type=ActionType.BASH_COMMAND,
                    command="ls -la /app 2>/dev/null || (mkdir -p /app && echo 'Created /app directory' && ls -la /app)",
                    reasoning="Verify or create the /app directory where files should be created"
                ),
                expected_outcome="/app directory listing or creation confirmation"
            ),
            Step(
                description="Display task description for reference",
                action=Action(
                    action_type=ActionType.BASH_COMMAND,
                    command=f"echo 'TASK: {task_description[:200]}...'",
                    reasoning="Keep task description visible for next planning phase"
                ),
                expected_outcome="Task description displayed"
            )
        ]

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from text, with aggressive cleanup.

        Args:
            text: Text potentially containing JSON

        Returns:
            Parsed JSON object (preferably an array)
        """
        import re

        logger.debug(f"Attempting to extract JSON from {len(text)} char response")

        # Strategy 1: Find the LARGEST JSON array (greedy matching)
        # Changed from r'\[[\s\S]*?\]' (non-greedy) to r'\[[\s\S]*\]' (greedy)
        array_matches = re.findall(r'\[[\s\S]*\]', text)

        # Sort by length descending to try largest first
        array_matches = sorted(array_matches, key=len, reverse=True)

        for i, json_str in enumerate(array_matches):
            try:
                # Try parsing with strict=False
                parsed = json.loads(json_str, strict=False)
                if isinstance(parsed, list):
                    logger.debug(f"Successfully extracted JSON array with {len(parsed)} elements (match #{i+1})")
                    return parsed
                logger.debug(f"Match #{i+1} parsed but not an array, skipping")
            except json.JSONDecodeError as e:
                # Try fixing common issues
                try:
                    # Fix unterminated strings by removing incomplete last element
                    fixed = re.sub(r',\s*\{[^}]*$', ']', json_str)
                    if not fixed.endswith(']'):
                        fixed += ']'
                    parsed = json.loads(fixed, strict=False)
                    if isinstance(parsed, list):
                        logger.debug(f"Fixed and extracted JSON array with {len(parsed)} elements (match #{i+1})")
                        return parsed
                except Exception as fix_error:
                    logger.debug(f"Match #{i+1} failed to parse: {e}, fix also failed: {fix_error}")
                    continue

        # Strategy 2: Look for JSON object (single step plan)
        obj_matches = re.findall(r'\{[\s\S]*\}', text)
        obj_matches = sorted(obj_matches, key=len, reverse=True)

        for i, json_str in enumerate(obj_matches):
            try:
                parsed = json.loads(json_str, strict=False)
                logger.debug(f"Extracted JSON object (will convert to single-item array) (match #{i+1})")
                return [parsed] if isinstance(parsed, dict) else parsed
            except Exception as e:
                logger.debug(f"Object match #{i+1} failed: {e}")
                continue

        # Strategy 3: Last resort - try parsing entire text
        try:
            parsed = json.loads(text.strip(), strict=False)
            logger.debug("Successfully parsed entire response as JSON")
            return parsed
        except Exception as e:
            logger.error(f"All JSON extraction strategies failed. Response preview: {text[:200]}...")
            raise ValueError(f"Could not extract valid JSON from text: {e}")

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
        # STRICT CHECK: If we've only executed a few commands, task CANNOT be complete
        if len(recent_observations) < 8:
            return False

        # Check if any actual implementation files were created
        # Be careful to NOT match stderr/stdout redirection like 2>/dev/null or >&2
        import re
        has_file_creation = any(
            # Match output redirection that creates files (not stderr/stdout redirection)
            re.search(r'(?<![012&])[>]{1,2}\s*[^\s&]', obs.action.command) or
            "cat >" in obs.action.command or
            "tee " in obs.action.command
            for obs in recent_observations
        )

        if not has_file_creation:
            return False

        prompt = f"""Task: {task_description}

Recent execution history (showing all {len(recent_observations)} steps):
{self._format_observations(recent_observations)}

Analyze if the task is FULLY COMPLETED. Be EXTREMELY STRICT.

The task is ONLY complete if ALL of these are true:
1. All required files have been created with actual implementation code (not just stubs)
2. The code has been tested and tests PASSED
3. All specific requirements from the task description are met
4. There is evidence of successful execution/testing in the output

Respond with JSON:
{{
    "is_complete": true or false,
    "evidence": "Specific evidence from the output showing completion (or what's missing)",
    "files_created": ["list of files that were created"],
    "tests_passed": true or false,
    "missing_requirements": ["list of requirements not yet met"]
}}

Be STRICT: If in doubt, the task is NOT complete."""

        try:
            response = await self.llm(prompt)
            data = self._extract_json(response)

            # Require explicit confirmation with supporting evidence
            is_complete = data.get("is_complete", False)
            tests_passed = data.get("tests_passed", False)
            files_created = data.get("files_created", [])
            missing = data.get("missing_requirements", [])

            # Only return True if:
            # 1. LLM says complete
            # 2. Tests passed
            # 3. At least one file was created
            # 4. No missing requirements
            return is_complete and tests_passed and len(files_created) > 0 and len(missing) == 0
        except Exception:
            # If we can't verify, assume NOT complete
            return False

    async def _replan(self, recent_observations: List[Observation]) -> None:
        """Create a new plan based on current progress.

        Args:
            recent_observations: Recent observations from execution
        """
        # Analyze what failed/succeeded in recent attempts
        failed_commands = [obs for obs in recent_observations[-10:] if not obs.success]
        successful_commands = [obs for obs in recent_observations[-10:] if obs.success]

        failure_summary = ""
        if failed_commands:
            failure_summary = "\n\nFailed approaches (DO NOT REPEAT THESE):\n"
            for obs in failed_commands[-5:]:
                failure_summary += f"- Command: {obs.action.command}\n  Error: {obs.error}\n"

        prompt = f"""Task: {self.task_description}

This is REPLAN #{self.replan_count + 1}. Previous plan(s) were incomplete.

Recent execution history:
{self._format_observations(recent_observations[-15:])}
{failure_summary}

CRITICAL INSTRUCTIONS FOR REPLANNING:
1. Analyze what FAILED in previous attempts (see above) and try a DIFFERENT approach
2. Build on what SUCCEEDED - don't redo successful steps
3. Focus on what's still MISSING to complete the task
4. If stuck on same error, try alternative methods (different tools, libraries, approaches)

What still needs to be done:
1. What files still need to be created or fixed?
2. What code needs to be written or corrected?
3. What tests need to pass?
4. What dependencies or configurations are missing?
5. What specific task requirements are not yet met?

Respond with a JSON array. Each step must have:
- description: Clear description of what this step does
- action_type: "bash" (for commands)
- command: The exact command to run with full paths
- expected_outcome: What should result from this step

Create your DETAILED, SPECIFIC continuation plan (5-10 steps). Begin with '[':"""

        try:
            response = await self.llm(prompt)
            steps = self._parse_plan_response(response)
            if steps:
                self.current_plan = steps
                self.current_step_index = 0
                logger.info(f"Replanned with {len(steps)} new steps")
            else:
                # Fallback to simple exploration
                logger.warning("Replanning returned empty plan, falling back to exploration")
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
        except Exception as e:
            # Fallback to simple exploration
            logger.error(f"Replanning failed: {type(e).__name__}: {e}")
            logger.warning("Falling back to exploration")
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
