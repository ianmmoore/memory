"""HaluMem benchmark runner.

Orchestrates the complete HaluMem benchmark evaluation including
extraction, updating, and question answering phases.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, asdict
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import HaluMemConfig, ModelConfig
from .dataset import HaluMemDataset
from .extraction import ExtractionEvaluator, create_llm_extraction_fn
from .updating import UpdateEvaluator, create_llm_update_fn, UpdateDecision
from .qa import QAEvaluator, create_llm_judge_fn
from .metrics import (
    ExtractionMetrics,
    UpdateMetrics,
    QAMetrics,
    aggregate_metrics
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Complete results from a HaluMem benchmark run.

    Attributes:
        config: Configuration used for this run.
        extraction: Extraction evaluation metrics.
        updating: Update evaluation metrics.
        qa: Question answering metrics.
        timing: Timing information for each phase.
        cost_estimate: Estimated API cost in USD.
    """
    config: Dict[str, Any]
    extraction: Dict[str, float]
    updating: Dict[str, Any]
    qa: Dict[str, Any]
    timing: Dict[str, float]
    cost_estimate: Dict[str, float]
    timestamp: str = ""
    run_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.run_id:
            self.run_id = f"halumem_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {path}")

    def print_summary(self) -> None:
        """Print a summary of the results."""
        print("\n" + "=" * 60)
        print("HaluMem Benchmark Results")
        print("=" * 60)
        print(f"Run ID: {self.run_id}")
        print(f"Variant: {self.config.get('variant', 'unknown')}")
        print(f"Prefilter: {'enabled' if self.config.get('prefilter_enabled') else 'disabled'}")
        print()

        print("EXTRACTION METRICS:")
        print(f"  Recall:          {self.extraction.get('recall', 0):.2%}")
        print(f"  Weighted Recall: {self.extraction.get('weighted_recall', 0):.2%}")
        print(f"  Precision:       {self.extraction.get('precision', 0):.2%}")
        print()

        print("UPDATE METRICS:")
        print(f"  Accuracy:        {self.updating.get('accuracy', 0):.2%}")
        print(f"  Omission Rate:   {self.updating.get('omission_rate', 0):.2%}")
        print(f"  Hallucination:   {self.updating.get('hallucination_rate', 0):.2%}")
        print()

        print("QA METRICS:")
        print(f"  Correctness:     {self.qa.get('correctness', 0):.2%}")
        print(f"  Hallucination:   {self.qa.get('hallucination_rate', 0):.2%}")
        print(f"  Omission:        {self.qa.get('omission_rate', 0):.2%}")
        print()

        print("TIMING:")
        for phase, seconds in self.timing.items():
            print(f"  {phase}: {seconds:.1f}s")
        print()

        print("ESTIMATED COST:")
        total = sum(self.cost_estimate.values())
        for component, cost in self.cost_estimate.items():
            print(f"  {component}: ${cost:.2f}")
        print(f"  TOTAL: ${total:.2f}")
        print("=" * 60 + "\n")


class HaluMemBenchmark:
    """Main benchmark orchestrator.

    This class coordinates all phases of the HaluMem benchmark:
    1. Dataset loading
    2. Memory extraction evaluation
    3. Memory update evaluation
    4. Question answering evaluation
    5. Results aggregation and reporting

    Example:
        >>> config = HaluMemConfig.for_long()
        >>> benchmark = HaluMemBenchmark(config)
        >>> await benchmark.setup()
        >>> results = await benchmark.run_full()
        >>> results.print_summary()
    """

    def __init__(self, config: HaluMemConfig):
        """Initialize the benchmark.

        Args:
            config: HaluMem configuration.
        """
        self.config = config
        self.dataset: Optional[HaluMemDataset] = None
        self.memory_system: Optional[Any] = None

        # Evaluators (created during setup)
        self.extraction_evaluator: Optional[ExtractionEvaluator] = None
        self.update_evaluator: Optional[UpdateEvaluator] = None
        self.qa_evaluator: Optional[QAEvaluator] = None

        # Model functions (set via configure_models)
        self.extraction_fn: Optional[Callable] = None
        self.update_fn: Optional[Callable] = None
        self.qa_fn: Optional[Callable] = None
        self.judge_fn: Optional[Callable] = None

    async def setup(self) -> None:
        """Set up the benchmark by loading dataset and initializing evaluators."""
        logger.info(f"Setting up HaluMem-{self.config.variant} benchmark...")

        # Load dataset
        self.dataset = HaluMemDataset(self.config)
        await self.dataset.load()

        # Initialize evaluators
        self.extraction_evaluator = ExtractionEvaluator(self.config, self.dataset)
        self.update_evaluator = UpdateEvaluator(self.config, self.dataset)
        self.qa_evaluator = QAEvaluator(self.config, self.dataset)

        stats = self.dataset.get_stats()
        logger.info(f"Dataset loaded: {stats}")

    def configure_models(
        self,
        extraction_model_fn: Callable[[str], Awaitable[str]],
        scoring_model_fn: Callable[[str], Awaitable[str]],
        primary_model_fn: Callable[[str], Awaitable[str]],
        judge_model_fn: Optional[Callable[[str], Awaitable[str]]] = None,
        embedding_fn: Optional[Callable[[List[str]], Awaitable[List[List[float]]]]] = None
    ) -> None:
        """Configure the model functions for the benchmark.

        Args:
            extraction_model_fn: Function for memory extraction (GPT-5.1).
            scoring_model_fn: Function for relevance scoring (GPT-5 nano).
            primary_model_fn: Function for QA responses (GPT-5.1).
            judge_model_fn: Function for answer judging. Defaults to primary_model_fn.
            embedding_fn: Optional function for embeddings.
        """
        # Create extraction function
        self.extraction_fn = create_llm_extraction_fn(extraction_model_fn)

        # Create update function
        self.update_fn = create_llm_update_fn(primary_model_fn)

        # Create judge function
        self.judge_fn = create_llm_judge_fn(judge_model_fn or primary_model_fn)

        # Store for QA setup
        self._scoring_model_fn = scoring_model_fn
        self._primary_model_fn = primary_model_fn
        self._embedding_fn = embedding_fn

        logger.info("Model functions configured")

    async def setup_memory_system(self) -> None:
        """Set up the memory system for QA evaluation."""
        from memory_lib import MemorySystem

        self.memory_system = MemorySystem(
            small_model_fn=self._scoring_model_fn,
            relevance_threshold=self.config.relevance_threshold,
            max_memories=self.config.max_memories,
            batch_size=self.config.batch_size,
            embedding_fn=self._embedding_fn,
            enable_prefilter=self.config.prefilter_enabled,
            prefilter_top_k=self.config.prefilter_top_k
        )

        logger.info("Memory system initialized")

    async def run_extraction(self) -> ExtractionMetrics:
        """Run the extraction evaluation phase."""
        if not self.extraction_evaluator or not self.extraction_fn:
            raise RuntimeError("Benchmark not set up. Call setup() and configure_models() first.")

        logger.info("Running extraction evaluation...")
        return await self.extraction_evaluator.evaluate(self.extraction_fn)

    async def run_updating(self) -> UpdateMetrics:
        """Run the update evaluation phase."""
        if not self.update_evaluator or not self.update_fn:
            raise RuntimeError("Benchmark not set up. Call setup() and configure_models() first.")

        logger.info("Running update evaluation...")
        return await self.update_evaluator.evaluate(self.update_fn)

    async def run_qa(self) -> QAMetrics:
        """Run the QA evaluation phase."""
        if not self.qa_evaluator or not self.judge_fn:
            raise RuntimeError("Benchmark not set up. Call setup() and configure_models() first.")

        if not self.memory_system:
            await self.setup_memory_system()

        # Load extracted memories into the system
        # (In a real benchmark, this would use the extracted memories from phase 1)
        for mem in self.dataset.get_memories().values():
            self.memory_system.add_memory(
                text=mem.text,
                metadata={"category": mem.category, "importance": mem.importance}
            )

        # Generate embeddings if prefiltering is enabled
        if self.config.prefilter_enabled and self._embedding_fn:
            await self.memory_system.generate_embeddings()

        # Create QA function using memory system
        async def qa_fn(question: str) -> tuple[str, List[str]]:
            memories = await self.memory_system.retrieve_relevant_memories(question)
            memory_ids = [m.memory_id for m in memories]

            memory_context = self.memory_system.format_memories_for_prompt(memories)

            prompt = f"""Based on the following memories, answer the question concisely.

MEMORIES:
{memory_context}

QUESTION: {question}

ANSWER:"""

            answer = await self._primary_model_fn(prompt)
            return answer.strip(), memory_ids

        logger.info("Running QA evaluation...")
        return await self.qa_evaluator.evaluate(qa_fn, self.judge_fn)

    async def run_full(self) -> BenchmarkResults:
        """Run the complete benchmark.

        Returns:
            BenchmarkResults with all metrics.
        """
        import time

        timing = {}
        cost_estimate = {}

        # Extraction phase
        start = time.time()
        extraction_metrics = await self.run_extraction()
        timing["extraction"] = time.time() - start
        cost_estimate["extraction"] = self._estimate_extraction_cost()

        # Update phase
        start = time.time()
        update_metrics = await self.run_updating()
        timing["updating"] = time.time() - start
        cost_estimate["updating"] = self._estimate_update_cost()

        # QA phase
        start = time.time()
        qa_metrics = await self.run_qa()
        timing["qa"] = time.time() - start
        cost_estimate["qa"] = self._estimate_qa_cost()

        results = BenchmarkResults(
            config={
                "variant": self.config.variant,
                "prefilter_enabled": self.config.prefilter_enabled,
                "prefilter_top_k": self.config.prefilter_top_k,
                "relevance_threshold": self.config.relevance_threshold,
                "max_memories": self.config.max_memories,
                "sample_size": self.config.sample_size
            },
            extraction=extraction_metrics.to_dict(),
            updating=update_metrics.to_dict(),
            qa=qa_metrics.to_dict(),
            timing=timing,
            cost_estimate=cost_estimate
        )

        # Save results
        output_path = self.config.output_path / f"{results.run_id}.json"
        results.save(output_path)

        return results

    def _estimate_extraction_cost(self) -> float:
        """Estimate extraction cost based on config and pricing."""
        # GPT-5.1 pricing: $1.25/1M input, $10/1M output
        if self.config.variant == "long":
            input_tokens = 1_000_000  # 1M tokens
        else:
            input_tokens = 160_000  # 160K tokens

        output_tokens = 50_000  # Estimated extracted memories

        input_cost = (input_tokens / 1_000_000) * 1.25
        output_cost = (output_tokens / 1_000_000) * 10.00

        return input_cost + output_cost

    def _estimate_update_cost(self) -> float:
        """Estimate update evaluation cost."""
        num_scenarios = len(self.dataset.get_update_scenarios()) if self.dataset else 500

        # Each scenario: ~500 tokens in, ~100 tokens out
        input_tokens = num_scenarios * 500
        output_tokens = num_scenarios * 100

        input_cost = (input_tokens / 1_000_000) * 1.25
        output_cost = (output_tokens / 1_000_000) * 10.00

        return input_cost + output_cost

    def _estimate_qa_cost(self) -> float:
        """Estimate QA evaluation cost."""
        num_qa = self.config.sample_size or (len(self.dataset.get_qa_pairs()) if self.dataset else 3500)

        # Scoring (GPT-5 nano): 100 candidates × num_qa × 350 tokens
        if self.config.prefilter_enabled:
            candidates_per_query = self.config.prefilter_top_k
        else:
            # Assume 24K memories without prefiltering
            candidates_per_query = 24000

        scoring_tokens = candidates_per_query * num_qa * 350
        scoring_cost = (scoring_tokens / 1_000_000) * 0.05  # nano input
        scoring_output = candidates_per_query * num_qa * 50
        scoring_cost += (scoring_output / 1_000_000) * 0.40  # nano output

        # Primary model (GPT-5.1): num_qa × 3000 tokens in, 500 out
        primary_input = num_qa * 3000
        primary_output = num_qa * 500
        primary_cost = (primary_input / 1_000_000) * 1.25
        primary_cost += (primary_output / 1_000_000) * 10.00

        # Judge (GPT-5.1): num_qa × 1000 tokens in, 100 out
        judge_input = num_qa * 1000
        judge_output = num_qa * 100
        judge_cost = (judge_input / 1_000_000) * 1.25
        judge_cost += (judge_output / 1_000_000) * 10.00

        return scoring_cost + primary_cost + judge_cost


async def run_benchmark(
    variant: str = "long",
    prefilter: bool = True,
    sample_size: Optional[int] = None,
    api_key: Optional[str] = None
) -> BenchmarkResults:
    """Convenience function to run the benchmark.

    Args:
        variant: "medium" or "long".
        prefilter: Whether to enable prefiltering.
        sample_size: Optional limit on QA pairs.
        api_key: OpenAI API key.

    Returns:
        BenchmarkResults.
    """
    import os
    import openai

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

    client = openai.AsyncOpenAI(api_key=api_key)

    # Create model functions
    # Using gpt-5-nano for scoring (fast/cheap) and gpt-5.1 for primary tasks
    async def small_model(prompt: str) -> str:
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content

    async def primary_model(prompt: str) -> str:
        response = await client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content

    async def embed(texts: List[str]) -> List[List[float]]:
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [item.embedding for item in response.data]

    # Configure benchmark
    config = HaluMemConfig(
        variant=variant,
        prefilter_enabled=prefilter,
        sample_size=sample_size
    )

    benchmark = HaluMemBenchmark(config)
    await benchmark.setup()

    benchmark.configure_models(
        extraction_model_fn=primary_model,
        scoring_model_fn=small_model,
        primary_model_fn=primary_model,
        embedding_fn=embed if prefilter else None
    )

    return await benchmark.run_full()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HaluMem benchmark")
    parser.add_argument("--variant", choices=["medium", "long"], default="long")
    parser.add_argument("--no-prefilter", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results = asyncio.run(run_benchmark(
        variant=args.variant,
        prefilter=not args.no_prefilter,
        sample_size=args.sample_size
    ))

    results.print_summary()
