"""Memory update evaluation for HaluMem benchmark.

This module evaluates the ability of a memory system to correctly
handle updates when new information conflicts with existing memories.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

from .config import HaluMemConfig
from .dataset import HaluMemDataset, UpdateScenario
from .metrics import UpdateMetrics, compute_update_metrics

logger = logging.getLogger(__name__)


class UpdateAction(Enum):
    """Possible update actions."""
    ADD = "ADD"          # Add new memory
    UPDATE = "UPDATE"    # Update existing memory
    DELETE = "DELETE"    # Delete memory
    NOOP = "NOOP"       # No operation needed


@dataclass
class UpdateDecision:
    """Result of an update decision.

    Attributes:
        scenario_id: ID of the update scenario.
        predicted_action: The action the system decided to take.
        predicted_result: The resulting memory text (if applicable).
        reasoning: Explanation for the decision.
    """
    scenario_id: str
    predicted_action: str
    predicted_result: Optional[str] = None
    reasoning: str = ""


class UpdateEvaluator:
    """Evaluates memory update capabilities.

    This class tests how well a memory system handles conflicts
    between existing and new information.

    Example:
        >>> evaluator = UpdateEvaluator(config, dataset)
        >>> metrics = await evaluator.evaluate(update_fn=my_update_function)
        >>> print(f"Accuracy: {metrics.accuracy:.2%}")
    """

    def __init__(
        self,
        config: HaluMemConfig,
        dataset: HaluMemDataset
    ):
        """Initialize the update evaluator.

        Args:
            config: HaluMem configuration.
            dataset: Loaded HaluMem dataset.
        """
        self.config = config
        self.dataset = dataset

    async def evaluate_scenario(
        self,
        scenario: UpdateScenario,
        update_fn: Callable[[str, str], Awaitable[UpdateDecision]]
    ) -> UpdateDecision:
        """Evaluate a single update scenario.

        Args:
            scenario: The update scenario to evaluate.
            update_fn: Async function that takes (old_memory, new_info) and
                returns an UpdateDecision.

        Returns:
            The update decision made by the system.
        """
        decision = await update_fn(scenario.old_memory, scenario.new_information)
        decision.scenario_id = scenario.scenario_id
        return decision

    async def evaluate(
        self,
        update_fn: Callable[[str, str], Awaitable[UpdateDecision]],
        scenario_ids: Optional[List[str]] = None
    ) -> UpdateMetrics:
        """Run update evaluation on the dataset.

        Args:
            update_fn: Async function that takes (old_memory, new_info) and
                returns an UpdateDecision.
            scenario_ids: Optional list of scenario IDs to evaluate.
                If None, evaluates all scenarios.

        Returns:
            UpdateMetrics with evaluation results.
        """
        scenarios = self.dataset.get_update_scenarios()

        if scenario_ids:
            scenarios = [s for s in scenarios if s.scenario_id in scenario_ids]

        if not scenarios:
            logger.warning("No update scenarios to evaluate")
            return UpdateMetrics()

        predictions = []
        ground_truth = []
        action_types = []
        batch_size = self.config.batch_size or 5  # Process 5 scenarios in parallel to avoid rate limits

        async def evaluate_one(scenario) -> UpdateDecision:
            """Evaluate a single update scenario."""
            return await self.evaluate_scenario(scenario, update_fn)

        # Process in batches
        for batch_start in range(0, len(scenarios), batch_size):
            batch_end = min(batch_start + batch_size, len(scenarios))
            batch = scenarios[batch_start:batch_end]

            logger.info(f"Evaluating update batch {batch_start//batch_size + 1}/{(len(scenarios) + batch_size - 1)//batch_size} (scenarios {batch_start+1}-{batch_end}/{len(scenarios)})")

            # Run batch in parallel
            tasks = [evaluate_one(s) for s in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, (scenario, result) in enumerate(zip(batch, results)):
                if isinstance(result, Exception):
                    logger.error(f"Error evaluating scenario {scenario.scenario_id}: {result}")
                    predictions.append("ERROR")
                else:
                    predictions.append(result.predicted_action)
                    # Log mismatches
                    if result.predicted_action != scenario.expected_action:
                        logger.debug(
                            f"Mismatch in {scenario.scenario_id}: "
                            f"predicted={result.predicted_action}, "
                            f"expected={scenario.expected_action}"
                        )
                ground_truth.append(scenario.expected_action)
                action_types.append(scenario.conflict_type)

            # Small delay between batches to avoid rate limiting
            if batch_end < len(scenarios):
                await asyncio.sleep(1.0)

        metrics = compute_update_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            action_types=action_types
        )

        logger.info(
            f"Update evaluation complete: Accuracy={metrics.accuracy:.2%}, "
            f"Omission={metrics.omission_rate:.2%}, "
            f"Hallucination={metrics.hallucination_rate:.2%}"
        )

        return metrics


def create_llm_update_fn(
    model_fn: Callable[[str], Awaitable[str]]
) -> Callable[[str, str], Awaitable[UpdateDecision]]:
    """Create an update decision function using an LLM.

    Args:
        model_fn: Async function to call the LLM.

    Returns:
        Async function that decides on memory updates.
    """
    UPDATE_PROMPT = """You are a memory management system. Given an existing memory and new information, decide what action to take.

EXISTING MEMORY:
{old_memory}

NEW INFORMATION:
{new_info}

Analyze if the new information:
1. CONTRADICTS the existing memory (requires UPDATE)
2. ADDS new information not in the existing memory (requires ADD)
3. Makes the existing memory obsolete (requires DELETE)
4. Is already captured or irrelevant (requires NOOP - no operation)

Respond in this exact format:
ACTION: <ADD|UPDATE|DELETE|NOOP>
RESULT: <new memory text if UPDATE or ADD, otherwise "N/A">
REASONING: <brief explanation>"""

    async def update_decision(old_memory: str, new_info: str) -> UpdateDecision:
        prompt = UPDATE_PROMPT.format(old_memory=old_memory, new_info=new_info)
        response = await model_fn(prompt)

        # Parse response
        action = "NOOP"
        result = None
        reasoning = ""

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("ACTION:"):
                action_str = line[7:].strip().upper()
                if action_str in ["ADD", "UPDATE", "DELETE", "NOOP"]:
                    action = action_str
            elif line.startswith("RESULT:"):
                result_str = line[7:].strip()
                if result_str.upper() != "N/A":
                    result = result_str
            elif line.startswith("REASONING:"):
                reasoning = line[10:].strip()

        return UpdateDecision(
            scenario_id="",  # Will be set by evaluator
            predicted_action=action,
            predicted_result=result,
            reasoning=reasoning
        )

    return update_decision
