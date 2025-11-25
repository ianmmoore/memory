"""HaluMem Benchmark - Evaluation suite for memory systems.

This module implements the HaluMem benchmark (arXiv:2511.03506) for evaluating
memory systems on extraction, updating, and question answering tasks.

Supports both HaluMem-Medium (~160K tokens) and HaluMem-Long (1M+ tokens).

Cost-optimized with embedding prefiltering, reducing evaluation cost from
~$3,000 to ~$65-100 for full HaluMem-Long benchmark.
"""

from .config import HaluMemConfig, ModelConfig
from .dataset import HaluMemDataset, DialogueTurn, MemoryPoint, QAPair, UpdateScenario
from .metrics import (
    ExtractionMetrics,
    UpdateMetrics,
    QAMetrics,
    compute_recall,
    compute_weighted_recall,
    compute_precision,
    compute_extraction_metrics,
    compute_update_metrics,
    compute_qa_metrics
)
from .extraction import ExtractionEvaluator, ExtractionResult, create_llm_extraction_fn
from .updating import UpdateEvaluator, UpdateDecision, create_llm_update_fn
from .qa import QAEvaluator, QAResult, create_llm_judge_fn
from .runner import HaluMemBenchmark, BenchmarkResults, run_benchmark

__all__ = [
    # Config
    "HaluMemConfig",
    "ModelConfig",
    # Dataset
    "HaluMemDataset",
    "DialogueTurn",
    "MemoryPoint",
    "QAPair",
    "UpdateScenario",
    # Metrics
    "ExtractionMetrics",
    "UpdateMetrics",
    "QAMetrics",
    "compute_recall",
    "compute_weighted_recall",
    "compute_precision",
    "compute_extraction_metrics",
    "compute_update_metrics",
    "compute_qa_metrics",
    # Evaluators
    "ExtractionEvaluator",
    "ExtractionResult",
    "UpdateEvaluator",
    "UpdateDecision",
    "QAEvaluator",
    "QAResult",
    # Functions
    "create_llm_extraction_fn",
    "create_llm_update_fn",
    "create_llm_judge_fn",
    # Runner
    "HaluMemBenchmark",
    "BenchmarkResults",
    "run_benchmark"
]
