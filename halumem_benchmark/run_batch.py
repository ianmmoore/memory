#!/usr/bin/env python3
"""Run HaluMem benchmark using OpenAI Batch API.

This version submits all requests as batch jobs, avoiding rate limits
and getting 50% cost savings.

Usage:
    python run_batch.py                    # Full benchmark
    python run_batch.py --sample-size 50   # Limited QA pairs
    python run_batch.py --phase extraction # Run single phase
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from halumem_benchmark import HaluMemConfig, HaluMemDataset
from halumem_benchmark.batch_api import BatchBenchmarkRunner
from halumem_benchmark.metrics import (
    compute_extraction_metrics,
    compute_update_metrics,
    compute_qa_metrics
)


async def run_batch_benchmark(
    variant: str = "long",
    sample_size: int = None,
    phase: str = None,
    output_dir: Path = None
):
    """Run the HaluMem benchmark using Batch API.

    Args:
        variant: "long" or "medium"
        sample_size: Limit QA pairs (for testing)
        phase: Run single phase ("extraction", "updating", "qa") or None for all
        output_dir: Where to save results
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    client = openai.AsyncOpenAI(api_key=api_key)

    # Setup
    config = HaluMemConfig.for_long() if variant == "long" else HaluMemConfig.for_medium()
    if sample_size:
        config.sample_size = sample_size

    output_dir = output_dir or Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"batch_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting batch benchmark run: {run_id}")

    # Load dataset
    dataset = HaluMemDataset(config)
    await dataset.load()
    stats = dataset.get_stats()
    logging.info(f"Dataset loaded: {stats}")

    # Initialize runner
    runner = BatchBenchmarkRunner(
        client=client,
        config=config,
        dataset=dataset,
        output_dir=run_dir
    )

    results = {
        "run_id": run_id,
        "variant": variant,
        "sample_size": sample_size,
        "started_at": datetime.now().isoformat(),
        "dataset_stats": stats
    }

    # Run phases SEQUENTIALLY (each phase feeds into the next)
    # This matches HaluMem's "three-step hallucination tracing mechanism"
    if phase is None:
        logging.info("=" * 60)
        logging.info("RUNNING SEQUENTIAL PIPELINE (apples-to-apples with SuperMemory)")
        logging.info("Extraction → Updating → QA")
        logging.info("=" * 60)

        total_start = time.time()
        ground_truth = dataset.get_memories()

        # ===== PHASE 1: EXTRACTION =====
        logging.info("\n" + "=" * 60)
        logging.info("PHASE 1: EXTRACTION")
        logging.info("=" * 60)

        start_time = time.time()
        extracted = await runner.run_extraction_batch()

        with open(run_dir / "extraction_raw.json", 'w') as f:
            json.dump(extracted, f, indent=2)

        # Build memory store from extracted memories
        memory_store = {}  # memory_id -> text
        all_extracted = []
        for dialogue_id, memories in extracted.items():
            for i, mem_text in enumerate(memories):
                mem_id = f"{dialogue_id}_m{i}"
                memory_store[mem_id] = mem_text
                all_extracted.append(mem_text)

        logging.info(f"Extracted {len(all_extracted)} memories into store")

        # Compute extraction metrics
        weights = {m.memory_id: m.importance for m in ground_truth.values()}
        matched_gt_ids = set()
        for ext_mem in all_extracted:
            ext_lower = ext_mem.lower()
            for gt_id, gt_mem in ground_truth.items():
                if ext_lower in gt_mem.text.lower() or gt_mem.text.lower() in ext_lower:
                    matched_gt_ids.add(gt_id)

        extraction_metrics = compute_extraction_metrics(
            extracted_memories=list(matched_gt_ids),
            ground_truth_memories=list(ground_truth.keys()),
            importance_weights=weights,
            target_memory_ids={m.memory_id for m in ground_truth.values() if m.is_target}
        )
        results["extraction"] = {
            "metrics": extraction_metrics.to_dict(),
            "time_seconds": time.time() - start_time,
            "total_extracted": len(all_extracted),
            "matched": len(matched_gt_ids)
        }
        logging.info(f"Extraction: Recall={extraction_metrics.recall:.2%}, Precision={extraction_metrics.precision:.2%}")

        # ===== PHASE 2: UPDATING =====
        logging.info("\n" + "=" * 60)
        logging.info("PHASE 2: UPDATING (applied to extracted memories)")
        logging.info("=" * 60)

        start_time = time.time()
        decisions = await runner.run_update_batch()

        with open(run_dir / "updating_raw.json", 'w') as f:
            json.dump(decisions, f, indent=2)

        # Apply updates to memory store
        scenarios = dataset.get_update_scenarios()
        updates_applied = 0
        for scenario in scenarios:
            decision = decisions.get(scenario.scenario_id, {"action": "NOOP"})
            if decision["action"] == "UPDATE" and decision.get("result"):
                # Find matching memory and update it
                for mem_id, mem_text in list(memory_store.items()):
                    if scenario.old_memory.lower() in mem_text.lower():
                        memory_store[mem_id] = decision["result"]
                        updates_applied += 1
                        break
            elif decision["action"] == "ADD" and decision.get("result"):
                new_id = f"update_{scenario.scenario_id}"
                memory_store[new_id] = decision["result"]
                updates_applied += 1
            elif decision["action"] == "DELETE":
                for mem_id, mem_text in list(memory_store.items()):
                    if scenario.old_memory.lower() in mem_text.lower():
                        del memory_store[mem_id]
                        updates_applied += 1
                        break

        logging.info(f"Applied {updates_applied} updates to memory store")
        logging.info(f"Memory store now has {len(memory_store)} memories")

        # Compute update metrics
        predictions = []
        ground_truth_actions = []
        action_types = []
        for scenario in scenarios:
            decision = decisions.get(scenario.scenario_id, {"action": "NOOP"})
            predictions.append(decision["action"])
            ground_truth_actions.append(scenario.expected_action)
            action_types.append(scenario.conflict_type)

        update_metrics = compute_update_metrics(
            predictions=predictions,
            ground_truth=ground_truth_actions,
            action_types=action_types
        )
        results["updating"] = {
            "metrics": update_metrics.to_dict(),
            "time_seconds": time.time() - start_time,
            "total_scenarios": len(scenarios),
            "updates_applied": updates_applied,
            "final_memory_count": len(memory_store)
        }
        logging.info(f"Updating: Accuracy={update_metrics.accuracy:.2%}")

        # ===== PHASE 3: QA (using final memory store) =====
        logging.info("\n" + "=" * 60)
        logging.info("PHASE 3: QA (using extracted + updated memories)")
        logging.info("=" * 60)

        start_time = time.time()

        # Use the memory store we built (extracted + updated), NOT ground truth
        qa_results = await runner.run_qa_batch(memory_store, sample_size=sample_size)

        with open(run_dir / "qa_raw.json", 'w') as f:
            json.dump(qa_results, f, indent=2)

        qa_pairs = dataset.get_qa_pairs(limit=sample_size)
        correctness_scores = []
        hallucination_flags = []
        omission_flags = []
        question_types = []
        for qa in qa_pairs:
            result = qa_results.get(qa.qa_id, {})
            correctness_scores.append(result.get("correctness", 0.0))
            hallucination_flags.append(result.get("hallucination", False))
            omission_flags.append(result.get("omission", False))
            question_types.append(qa.question_type)

        qa_metrics = compute_qa_metrics(
            correctness_scores=correctness_scores,
            hallucination_flags=hallucination_flags,
            omission_flags=omission_flags,
            question_types=question_types
        )
        results["qa"] = {
            "metrics": qa_metrics.to_dict(),
            "time_seconds": time.time() - start_time,
            "total_qa_pairs": len(qa_pairs),
            "memories_used": len(memory_store)
        }
        logging.info(f"QA: Correctness={qa_metrics.correctness:.2%}, Hallucination={qa_metrics.hallucination_rate:.2%}")

        results["total_time_seconds"] = time.time() - total_start

    # Single phase mode
    elif phase == "extraction":
        logging.info("=" * 60)
        logging.info("PHASE: EXTRACTION (Batch)")
        logging.info("=" * 60)

        start_time = time.time()
        extracted = await runner.run_extraction_batch()

        with open(run_dir / "extraction_raw.json", 'w') as f:
            json.dump(extracted, f, indent=2)

        all_extracted = []
        for memories in extracted.values():
            all_extracted.extend(memories)

        ground_truth = dataset.get_memories()
        weights = {m.memory_id: m.importance for m in ground_truth.values()}

        matched_gt_ids = set()
        for ext_mem in all_extracted:
            ext_lower = ext_mem.lower()
            for gt_id, gt_mem in ground_truth.items():
                if ext_lower in gt_mem.text.lower() or gt_mem.text.lower() in ext_lower:
                    matched_gt_ids.add(gt_id)

        extraction_metrics = compute_extraction_metrics(
            extracted_memories=list(matched_gt_ids),
            ground_truth_memories=list(ground_truth.keys()),
            importance_weights=weights,
            target_memory_ids={m.memory_id for m in ground_truth.values() if m.is_target}
        )

        results["extraction"] = {
            "metrics": extraction_metrics.to_dict(),
            "time_seconds": time.time() - start_time,
            "total_extracted": len(all_extracted),
            "matched": len(matched_gt_ids)
        }
        logging.info(f"Extraction complete: {extraction_metrics}")

    elif phase == "updating":
        logging.info("=" * 60)
        logging.info("PHASE: UPDATING (Batch)")
        logging.info("=" * 60)

        start_time = time.time()
        decisions = await runner.run_update_batch()

        with open(run_dir / "updating_raw.json", 'w') as f:
            json.dump(decisions, f, indent=2)

        scenarios = dataset.get_update_scenarios()
        predictions = []
        ground_truth_actions = []
        action_types = []

        for scenario in scenarios:
            decision = decisions.get(scenario.scenario_id, {"action": "NOOP"})
            predictions.append(decision["action"])
            ground_truth_actions.append(scenario.expected_action)
            action_types.append(scenario.conflict_type)

        update_metrics = compute_update_metrics(
            predictions=predictions,
            ground_truth=ground_truth_actions,
            action_types=action_types
        )

        results["updating"] = {
            "metrics": update_metrics.to_dict(),
            "time_seconds": time.time() - start_time,
            "total_scenarios": len(scenarios)
        }
        logging.info(f"Updating complete: {update_metrics}")

    elif phase == "qa":
        logging.info("=" * 60)
        logging.info("PHASE: QA (Batch)")
        logging.info("=" * 60)

        ground_truth = dataset.get_memories()
        memories = {m.memory_id: m.text for m in ground_truth.values()}

        start_time = time.time()
        qa_results = await runner.run_qa_batch(memories, sample_size=sample_size)

        with open(run_dir / "qa_raw.json", 'w') as f:
            json.dump(qa_results, f, indent=2)

        qa_pairs = dataset.get_qa_pairs(limit=sample_size)
        correctness_scores = []
        hallucination_flags = []
        omission_flags = []
        question_types = []

        for qa in qa_pairs:
            result = qa_results.get(qa.qa_id, {})
            correctness_scores.append(result.get("correctness", 0.0))
            hallucination_flags.append(result.get("hallucination", False))
            omission_flags.append(result.get("omission", False))
            question_types.append(qa.question_type)

        qa_metrics = compute_qa_metrics(
            correctness_scores=correctness_scores,
            hallucination_flags=hallucination_flags,
            omission_flags=omission_flags,
            question_types=question_types
        )

        results["qa"] = {
            "metrics": qa_metrics.to_dict(),
            "time_seconds": time.time() - start_time,
            "total_qa_pairs": len(qa_pairs)
        }
        logging.info(f"QA complete: {qa_metrics}")

    # Save final results
    results["completed_at"] = datetime.now().isoformat()

    with open(run_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("HALUMEM BATCH BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Variant: {variant}")
    print(f"Output: {run_dir}")
    print()

    if "extraction" in results:
        m = results["extraction"]["metrics"]
        print("EXTRACTION:")
        print(f"  Recall: {m.get('recall', 0):.2%}")
        print(f"  Weighted Recall: {m.get('weighted_recall', 0):.2%}")
        print(f"  Precision: {m.get('precision', 0):.2%}")
        print()

    if "updating" in results:
        m = results["updating"]["metrics"]
        print("UPDATING:")
        print(f"  Accuracy: {m.get('accuracy', 0):.2%}")
        print(f"  Omission Rate: {m.get('omission_rate', 0):.2%}")
        print(f"  Hallucination Rate: {m.get('hallucination_rate', 0):.2%}")
        print()

    if "qa" in results:
        m = results["qa"]["metrics"]
        print("QA:")
        print(f"  Correctness: {m.get('correctness', 0):.2%}")
        print(f"  Hallucination Rate: {m.get('hallucination_rate', 0):.2%}")
        print(f"  Omission Rate: {m.get('omission_rate', 0):.2%}")
        print()

    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run HaluMem benchmark using Batch API"
    )
    parser.add_argument(
        "--variant",
        choices=["long", "medium"],
        default="long"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit QA pairs for testing"
    )
    parser.add_argument(
        "--phase",
        choices=["extraction", "updating", "qa"],
        default=None,
        help="Run single phase only"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true"
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    asyncio.run(run_batch_benchmark(
        variant=args.variant,
        sample_size=args.sample_size,
        phase=args.phase
    ))


if __name__ == "__main__":
    main()
