"""Metrics calculation for HaluMem benchmark.

Implements the core metrics from the HaluMem paper (arXiv:2511.03506):
- Memory Recall / Weighted Recall
- Target Memory Precision
- QA Correctness / Hallucination Rate / Omission Rate
- Update Correctness metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
import numpy as np


@dataclass
class ExtractionMetrics:
    """Metrics for memory extraction evaluation.

    Attributes:
        recall: Fraction of ground truth memories that were extracted.
        weighted_recall: Recall weighted by memory importance.
        precision: Fraction of extracted memories that are correct.
        target_precision: Precision on target memories specifically.
        num_extracted: Number of memories extracted.
        num_ground_truth: Number of ground truth memories.
        hallucination_rate: Fraction of extracted memories not in ground truth.
    """
    recall: float = 0.0
    weighted_recall: float = 0.0
    precision: float = 0.0
    target_precision: float = 0.0
    num_extracted: int = 0
    num_ground_truth: int = 0
    hallucination_rate: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "recall": self.recall,
            "weighted_recall": self.weighted_recall,
            "precision": self.precision,
            "target_precision": self.target_precision,
            "num_extracted": self.num_extracted,
            "num_ground_truth": self.num_ground_truth,
            "hallucination_rate": self.hallucination_rate
        }


@dataclass
class UpdateMetrics:
    """Metrics for memory update evaluation.

    Attributes:
        accuracy: Fraction of correct update decisions.
        omission_rate: Fraction of updates that were missed.
        hallucination_rate: Fraction of false updates.
        action_accuracy: Accuracy broken down by action type.
    """
    accuracy: float = 0.0
    omission_rate: float = 0.0
    hallucination_rate: float = 0.0
    action_accuracy: Dict[str, float] = field(default_factory=dict)
    num_scenarios: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "omission_rate": self.omission_rate,
            "hallucination_rate": self.hallucination_rate,
            "action_accuracy": self.action_accuracy,
            "num_scenarios": self.num_scenarios
        }


@dataclass
class QAMetrics:
    """Metrics for question answering evaluation.

    Attributes:
        correctness: Fraction of questions answered correctly.
        hallucination_rate: Fraction of answers with hallucinated information.
        omission_rate: Fraction of answers missing key information.
        by_question_type: Metrics broken down by question type.
    """
    correctness: float = 0.0
    hallucination_rate: float = 0.0
    omission_rate: float = 0.0
    by_question_type: Dict[str, float] = field(default_factory=dict)
    num_questions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "correctness": self.correctness,
            "hallucination_rate": self.hallucination_rate,
            "omission_rate": self.omission_rate,
            "by_question_type": self.by_question_type,
            "num_questions": self.num_questions
        }


def compute_recall(
    extracted: Set[str],
    ground_truth: Set[str]
) -> float:
    """Compute recall: fraction of ground truth memories extracted.

    Args:
        extracted: Set of extracted memory IDs (or normalized text).
        ground_truth: Set of ground truth memory IDs (or normalized text).

    Returns:
        Recall score between 0 and 1.
    """
    if not ground_truth:
        return 1.0 if not extracted else 0.0

    intersection = extracted.intersection(ground_truth)
    return len(intersection) / len(ground_truth)


def compute_weighted_recall(
    extracted: Set[str],
    ground_truth: Set[str],
    weights: Dict[str, float]
) -> float:
    """Compute weighted recall, accounting for memory importance.

    Args:
        extracted: Set of extracted memory IDs.
        ground_truth: Set of ground truth memory IDs.
        weights: Dictionary mapping memory IDs to importance weights.

    Returns:
        Weighted recall score between 0 and 1.
    """
    if not ground_truth:
        return 1.0 if not extracted else 0.0

    total_weight = sum(weights.get(m, 1.0) for m in ground_truth)
    if total_weight == 0:
        return 0.0

    matched_weight = sum(
        weights.get(m, 1.0)
        for m in extracted.intersection(ground_truth)
    )

    return matched_weight / total_weight


def compute_precision(
    extracted: Set[str],
    ground_truth: Set[str]
) -> float:
    """Compute precision: fraction of extracted memories that are correct.

    Args:
        extracted: Set of extracted memory IDs.
        ground_truth: Set of ground truth memory IDs.

    Returns:
        Precision score between 0 and 1.
    """
    if not extracted:
        return 1.0  # No extractions = no false positives

    intersection = extracted.intersection(ground_truth)
    return len(intersection) / len(extracted)


def compute_f1(recall: float, precision: float) -> float:
    """Compute F1 score from recall and precision.

    Args:
        recall: Recall score.
        precision: Precision score.

    Returns:
        F1 score between 0 and 1.
    """
    if recall + precision == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_extraction_metrics(
    extracted_memories: List[str],
    ground_truth_memories: List[str],
    importance_weights: Dict[str, float],
    target_memory_ids: Optional[Set[str]] = None
) -> ExtractionMetrics:
    """Compute all extraction metrics.

    Args:
        extracted_memories: List of extracted memory texts or IDs.
        ground_truth_memories: List of ground truth memory texts or IDs.
        importance_weights: Mapping of memory ID to importance weight.
        target_memory_ids: Optional set of target memory IDs for target precision.

    Returns:
        ExtractionMetrics with all computed values.
    """
    extracted_set = set(extracted_memories)
    ground_truth_set = set(ground_truth_memories)

    recall = compute_recall(extracted_set, ground_truth_set)
    weighted_recall = compute_weighted_recall(extracted_set, ground_truth_set, importance_weights)
    precision = compute_precision(extracted_set, ground_truth_set)

    # Target precision (only on target memories)
    target_precision = 0.0
    if target_memory_ids:
        target_extracted = extracted_set.intersection(target_memory_ids)
        target_gt = ground_truth_set.intersection(target_memory_ids)
        target_precision = compute_precision(target_extracted, target_gt)

    # Hallucination rate (extracted but not in ground truth)
    false_positives = extracted_set - ground_truth_set
    hallucination_rate = len(false_positives) / len(extracted_set) if extracted_set else 0.0

    return ExtractionMetrics(
        recall=recall,
        weighted_recall=weighted_recall,
        precision=precision,
        target_precision=target_precision,
        num_extracted=len(extracted_set),
        num_ground_truth=len(ground_truth_set),
        hallucination_rate=hallucination_rate
    )


def compute_update_metrics(
    predictions: List[str],
    ground_truth: List[str],
    action_types: Optional[List[str]] = None
) -> UpdateMetrics:
    """Compute update evaluation metrics.

    Args:
        predictions: List of predicted actions/results.
        ground_truth: List of expected actions/results.
        action_types: Optional list of action types for breakdown.

    Returns:
        UpdateMetrics with all computed values.
    """
    if not ground_truth:
        return UpdateMetrics()

    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(ground_truth)

    # Compute omission rate (predictions that are NOOP when they shouldn't be)
    omissions = sum(
        1 for p, g in zip(predictions, ground_truth)
        if p == "NOOP" and g != "NOOP"
    )
    omission_rate = omissions / len(ground_truth)

    # Compute hallucination rate (predictions that are actions when they should be NOOP)
    hallucinations = sum(
        1 for p, g in zip(predictions, ground_truth)
        if p != "NOOP" and g == "NOOP"
    )
    hallucination_rate = hallucinations / len(ground_truth)

    # Action-level accuracy
    action_accuracy = {}
    if action_types:
        for action in set(action_types):
            indices = [i for i, a in enumerate(action_types) if a == action]
            if indices:
                action_correct = sum(1 for i in indices if predictions[i] == ground_truth[i])
                action_accuracy[action] = action_correct / len(indices)

    return UpdateMetrics(
        accuracy=accuracy,
        omission_rate=omission_rate,
        hallucination_rate=hallucination_rate,
        action_accuracy=action_accuracy,
        num_scenarios=len(ground_truth)
    )


def compute_qa_metrics(
    correctness_scores: List[float],
    hallucination_flags: List[bool],
    omission_flags: List[bool],
    question_types: Optional[List[str]] = None
) -> QAMetrics:
    """Compute QA evaluation metrics.

    Args:
        correctness_scores: List of correctness scores (0-1) for each answer.
        hallucination_flags: List of booleans indicating hallucination.
        omission_flags: List of booleans indicating omission.
        question_types: Optional list of question types for breakdown.

    Returns:
        QAMetrics with all computed values.
    """
    if not correctness_scores:
        return QAMetrics()

    n = len(correctness_scores)
    correctness = np.mean(correctness_scores)
    hallucination_rate = sum(hallucination_flags) / n
    omission_rate = sum(omission_flags) / n

    # By question type
    by_question_type = {}
    if question_types:
        for qt in set(question_types):
            indices = [i for i, t in enumerate(question_types) if t == qt]
            if indices:
                qt_scores = [correctness_scores[i] for i in indices]
                by_question_type[qt] = float(np.mean(qt_scores))

    return QAMetrics(
        correctness=float(correctness),
        hallucination_rate=hallucination_rate,
        omission_rate=omission_rate,
        by_question_type=by_question_type,
        num_questions=n
    )


def aggregate_metrics(
    metrics_list: List[ExtractionMetrics | UpdateMetrics | QAMetrics]
) -> Dict[str, float]:
    """Aggregate metrics from multiple runs for mean ± std.

    Args:
        metrics_list: List of metrics from multiple runs.

    Returns:
        Dictionary with mean and std for each metric.
    """
    if not metrics_list:
        return {}

    # Get all numeric fields
    result = {}
    sample = metrics_list[0].to_dict()

    for key, value in sample.items():
        if isinstance(value, (int, float)):
            values = [m.to_dict()[key] for m in metrics_list]
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))

    return result
