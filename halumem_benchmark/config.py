"""Configuration for HaluMem benchmark."""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for LLM models used in the benchmark.

    Attributes:
        extraction_model: Model for extracting memories from dialogues.
        scoring_model: Small model for relevance scoring (GPT-5 nano).
        primary_model: Model for QA responses (GPT-5.1).
        judge_model: Model for evaluating answer correctness.
        embedding_model: Model for generating embeddings.
    """
    extraction_model: str = "gpt-5.1"
    scoring_model: str = "gpt-5-nano"
    primary_model: str = "gpt-5.1"
    judge_model: str = "gpt-5.1"
    embedding_model: str = "text-embedding-3-small"

    # API configuration
    api_key: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class HaluMemConfig:
    """Configuration for HaluMem benchmark execution.

    Attributes:
        variant: Either "medium" (~160K tokens) or "long" (1M+ tokens).
        dataset_path: Path to the HaluMem dataset files.
        output_path: Path for benchmark results.
        models: Model configuration.
        prefilter_enabled: Whether to use embedding prefiltering.
        prefilter_top_k: Candidates to select before LLM scoring.
        num_runs: Number of evaluation runs for statistical significance.
        sample_size: Optional limit on QA pairs to evaluate (for cost control).
    """
    variant: str = "long"
    dataset_path: Path = field(default_factory=lambda: Path("./data/halumem"))
    output_path: Path = field(default_factory=lambda: Path("./results/halumem"))
    models: ModelConfig = field(default_factory=ModelConfig)

    # Memory system configuration
    relevance_threshold: float = 0.7
    max_memories: int = 50
    batch_size: int = 20

    # Prefiltering (cost optimization)
    prefilter_enabled: bool = True
    prefilter_top_k: int = 100

    # Evaluation configuration
    num_runs: int = 3  # For mean ± std calculation
    sample_size: Optional[int] = None  # None = use all QA pairs

    # Extraction configuration
    extraction_chunk_size: int = 50000  # Tokens per chunk for 1M+ processing

    def __post_init__(self):
        """Ensure paths are Path objects."""
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

    @classmethod
    def for_medium(cls, **kwargs) -> "HaluMemConfig":
        """Create configuration for HaluMem-Medium benchmark."""
        return cls(variant="medium", **kwargs)

    @classmethod
    def for_long(cls, **kwargs) -> "HaluMemConfig":
        """Create configuration for HaluMem-Long benchmark."""
        return cls(variant="long", **kwargs)

    @classmethod
    def for_quick_test(cls, **kwargs) -> "HaluMemConfig":
        """Create configuration for quick testing (limited samples)."""
        return cls(
            variant="medium",
            sample_size=100,  # Only 100 QA pairs
            num_runs=1,
            **kwargs
        )

    def get_dataset_files(self) -> dict:
        """Get paths to dataset files based on variant."""
        base = self.dataset_path / self.variant
        return {
            "dialogues": base / "dialogues.jsonl",
            "memories": base / "ground_truth_memories.jsonl",
            "qa_pairs": base / "qa_pairs.jsonl",
            "updates": base / "update_scenarios.jsonl",
            "importance_weights": base / "importance_weights.json"
        }
