"""Memory extraction evaluation for HaluMem benchmark.

This module evaluates the ability of a memory system to extract
relevant memories from long dialogues.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass

from .config import HaluMemConfig
from .dataset import HaluMemDataset, DialogueTurn, MemoryPoint
from .metrics import ExtractionMetrics, compute_extraction_metrics

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of memory extraction from a dialogue.

    Attributes:
        dialogue_idx: Index of the source dialogue.
        extracted_memories: List of extracted memory texts.
        extraction_time_seconds: Time taken for extraction.
        token_count: Approximate token count of the dialogue.
    """
    dialogue_idx: int
    extracted_memories: List[str]
    extraction_time_seconds: float
    token_count: int = 0


class ExtractionEvaluator:
    """Evaluates memory extraction capabilities.

    This class processes dialogues through a memory system and compares
    the extracted memories against ground truth.

    Example:
        >>> evaluator = ExtractionEvaluator(config, dataset)
        >>> metrics = await evaluator.evaluate(
        ...     extraction_fn=my_extraction_function,
        ...     memory_system=my_memory_system
        ... )
        >>> print(f"Recall: {metrics.recall:.2%}")
    """

    def __init__(
        self,
        config: HaluMemConfig,
        dataset: HaluMemDataset
    ):
        """Initialize the extraction evaluator.

        Args:
            config: HaluMem configuration.
            dataset: Loaded HaluMem dataset.
        """
        self.config = config
        self.dataset = dataset

    async def extract_memories_from_dialogue(
        self,
        dialogue: List[DialogueTurn],
        extraction_fn: Callable[[str], Awaitable[List[str]]],
        chunk_size: Optional[int] = None
    ) -> List[str]:
        """Extract memories from a dialogue using the provided function.

        For long dialogues (1M+ tokens), this chunks the dialogue and
        processes each chunk, then deduplicates the results.

        Args:
            dialogue: List of dialogue turns.
            extraction_fn: Async function that takes dialogue text and returns
                a list of extracted memory strings.
            chunk_size: Optional chunk size in characters. Uses config if None.

        Returns:
            List of extracted memory strings.
        """
        # Format dialogue as text
        dialogue_text = self._format_dialogue(dialogue)

        # Estimate tokens (rough: 4 chars per token)
        estimated_tokens = len(dialogue_text) // 4
        chunk_size = chunk_size or self.config.extraction_chunk_size

        # If small enough, process in one go
        if estimated_tokens <= chunk_size:
            return await extraction_fn(dialogue_text)

        # Chunk and process
        all_memories = []
        chunks = self._chunk_text(dialogue_text, chunk_size * 4)  # Convert tokens to chars

        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_memories = await extraction_fn(chunk)
            all_memories.extend(chunk_memories)

        # Deduplicate (simple approach - exact match)
        return list(set(all_memories))

    def _format_dialogue(self, dialogue: List[DialogueTurn]) -> str:
        """Format dialogue turns into text."""
        lines = []
        for turn in dialogue:
            role_label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content}")
        return "\n\n".join(lines)

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks, trying to break at paragraph boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # Find end of chunk
            end_pos = min(current_pos + chunk_size, len(text))

            # Try to break at a paragraph boundary
            if end_pos < len(text):
                # Look for double newline within last 20% of chunk
                search_start = int(end_pos - chunk_size * 0.2)
                para_break = text.rfind("\n\n", search_start, end_pos)
                if para_break > current_pos:
                    end_pos = para_break + 2

            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos

        return chunks

    def match_memories(
        self,
        extracted: List[str],
        ground_truth: Dict[str, MemoryPoint],
        similarity_threshold: float = 0.8
    ) -> Dict[str, str]:
        """Match extracted memories to ground truth.

        Uses semantic similarity to match extracted memories to ground truth.
        For simplicity, this uses basic string matching. A more sophisticated
        implementation would use embeddings.

        Args:
            extracted: List of extracted memory strings.
            ground_truth: Dictionary of ground truth memories.
            similarity_threshold: Minimum similarity for a match.

        Returns:
            Dictionary mapping extracted memory to matched ground truth ID.
        """
        matches = {}

        for ext_mem in extracted:
            ext_lower = ext_mem.lower().strip()
            best_match = None
            best_score = 0.0

            for mem_id, gt_mem in ground_truth.items():
                gt_lower = gt_mem.text.lower().strip()

                # Simple substring matching
                if ext_lower in gt_lower or gt_lower in ext_lower:
                    score = 1.0
                else:
                    # Jaccard similarity on words
                    ext_words = set(ext_lower.split())
                    gt_words = set(gt_lower.split())
                    if ext_words and gt_words:
                        intersection = len(ext_words & gt_words)
                        union = len(ext_words | gt_words)
                        score = intersection / union
                    else:
                        score = 0.0

                if score > best_score and score >= similarity_threshold:
                    best_score = score
                    best_match = mem_id

            if best_match:
                matches[ext_mem] = best_match

        return matches

    async def evaluate(
        self,
        extraction_fn: Callable[[str], Awaitable[List[str]]],
        dialogue_indices: Optional[List[int]] = None
    ) -> ExtractionMetrics:
        """Run extraction evaluation on the dataset.

        Args:
            extraction_fn: Async function that takes dialogue text and returns
                a list of extracted memory strings.
            dialogue_indices: Optional list of dialogue indices to evaluate.
                If None, evaluates all dialogues.

        Returns:
            ExtractionMetrics with evaluation results.
        """
        import time

        dialogues = self.dataset.get_dialogues()
        ground_truth = self.dataset.get_memories()

        if dialogue_indices:
            dialogues = [dialogues[i] for i in dialogue_indices if i < len(dialogues)]

        if not dialogues:
            logger.warning("No dialogues to evaluate")
            return ExtractionMetrics()

        all_extracted = []
        total_time = 0.0
        batch_size = self.config.batch_size or 5  # Process 5 dialogues in parallel to avoid rate limits

        async def extract_one(idx: int, dialogue: List) -> List[str]:
            """Extract memories from a single dialogue."""
            return await self.extract_memories_from_dialogue(dialogue, extraction_fn)

        # Process in batches
        for batch_start in range(0, len(dialogues), batch_size):
            batch_end = min(batch_start + batch_size, len(dialogues))
            batch = dialogues[batch_start:batch_end]

            logger.info(f"Extracting batch {batch_start//batch_size + 1}/{(len(dialogues) + batch_size - 1)//batch_size} (dialogues {batch_start+1}-{batch_end}/{len(dialogues)})")
            start_time = time.time()

            # Run batch in parallel
            tasks = [extract_one(batch_start + i, d) for i, d in enumerate(batch)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error extracting dialogue {batch_start + i + 1}: {result}")
                else:
                    all_extracted.extend(result)

            elapsed = time.time() - start_time
            total_time += elapsed
            logger.info(f"Batch complete in {elapsed:.2f}s")

            # Small delay between batches to avoid rate limiting
            if batch_end < len(dialogues):
                await asyncio.sleep(1.0)

        # Match extracted to ground truth
        matches = self.match_memories(all_extracted, ground_truth)

        # Get matched ground truth IDs
        matched_gt_ids = set(matches.values())
        all_gt_ids = set(ground_truth.keys())

        # Get importance weights
        weights = {
            mem_id: self.dataset.get_importance_weight(mem_id)
            for mem_id in all_gt_ids
        }

        # Compute metrics
        metrics = compute_extraction_metrics(
            extracted_memories=list(matched_gt_ids),
            ground_truth_memories=list(all_gt_ids),
            importance_weights=weights,
            target_memory_ids={m.memory_id for m in ground_truth.values() if m.is_target}
        )

        logger.info(
            f"Extraction complete: Recall={metrics.recall:.2%}, "
            f"Weighted Recall={metrics.weighted_recall:.2%}, "
            f"Precision={metrics.precision:.2%}"
        )

        return metrics


def create_llm_extraction_fn(
    model_fn: Callable[[str], Awaitable[str]],
    model_name: str = "gpt-5.1"
) -> Callable[[str], Awaitable[List[str]]]:
    """Create an extraction function using an LLM.

    Args:
        model_fn: Async function to call the LLM.
        model_name: Name of the model for logging.

    Returns:
        Async function that extracts memories from dialogue text.
    """
    EXTRACTION_PROMPT = """You are a memory extraction system. Your task is to identify and extract key facts, preferences, and information from the following conversation that would be useful to remember for future interactions.

Extract discrete, atomic facts that:
1. Describe the user's personal information (name, job, location, etc.)
2. Capture user preferences and opinions
3. Record important events or experiences mentioned
4. Note any commitments, goals, or plans

Format each memory as a single concise sentence. Output one memory per line.

CONVERSATION:
{dialogue}

EXTRACTED MEMORIES (one per line):"""

    async def extract(dialogue_text: str) -> List[str]:
        prompt = EXTRACTION_PROMPT.format(dialogue=dialogue_text)
        response = await model_fn(prompt)

        # Parse response - each line is a memory
        memories = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Skip empty lines and numbered prefixes
            if line and not line.startswith("#"):
                # Remove bullet points or numbers
                if line[0] in "-•*" or (line[0].isdigit() and line[1:2] in ".):"):
                    line = line[2:].strip()
                if line:
                    memories.append(line)

        return memories

    return extract
