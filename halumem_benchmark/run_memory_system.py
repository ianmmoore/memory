#!/usr/bin/env python3
"""Run HaluMem benchmark using our actual MemorySystem implementation.

This tests our full memory system stack:
- Extraction with our prompts
- Relevance scoring with gpt-5-nano
- Prefiltering with embeddings
- Update/conflict detection
- SQLite storage

Usage:
    python run_memory_system.py                    # Full benchmark
    python run_memory_system.py --sample-size 50   # Limited QA pairs
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
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from memory_lib.general import MemorySystem, create_openai_embedding_fn
from halumem_benchmark import HaluMemConfig, HaluMemDataset
from halumem_benchmark.metrics import (
    compute_extraction_metrics,
    compute_update_metrics,
    compute_qa_metrics
)

logger = logging.getLogger(__name__)


async def run_memory_system_benchmark(
    variant: str = "long",
    sample_size: int = None,
    output_dir: Path = None,
    enable_prefilter: bool = True
):
    """Run HaluMem benchmark using our MemorySystem.

    This is an apples-to-apples comparison with SuperMemory, Mem0, etc.
    because we're testing our actual implementation, not raw LLM calls.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    client = openai.AsyncOpenAI(api_key=api_key)

    # Create model functions
    async def gpt5_nano(prompt: str) -> str:
        """Small model for relevance scoring."""
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content

    async def gpt5_1(prompt: str) -> str:
        """Primary model for extraction, updates, QA."""
        response = await client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content

    async def embed_fn(texts: List[str]) -> List[List[float]]:
        """Embedding function for prefiltering."""
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [item.embedding for item in response.data]

    # Setup
    config = HaluMemConfig.for_long() if variant == "long" else HaluMemConfig.for_medium()
    if sample_size:
        config.sample_size = sample_size

    output_dir = output_dir or Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"memsys_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize our MemorySystem
    db_path = str(run_dir / "memories.db")

    # Prefilter: top 1000 or top 10% (whichever is more)
    # We'll update this after extraction when we know memory count
    initial_top_k = 1000

    memory_system = MemorySystem(
        small_model_fn=gpt5_nano,
        db_path=db_path,
        relevance_threshold=0.5,  # At least 0.5 threshold
        max_memories=20,
        batch_size=5,  # Parallel scoring calls
        embedding_fn=embed_fn if enable_prefilter else None,
        enable_prefilter=enable_prefilter,
        prefilter_top_k=initial_top_k
    )

    logger.info(f"Initialized MemorySystem: db={db_path}, prefilter={enable_prefilter}")

    # Load dataset
    dataset = HaluMemDataset(config)
    await dataset.load()
    stats = dataset.get_stats()
    logger.info(f"Dataset loaded: {stats}")

    results = {
        "run_id": run_id,
        "variant": variant,
        "sample_size": sample_size,
        "prefilter_enabled": enable_prefilter,
        "started_at": datetime.now().isoformat(),
        "dataset_stats": stats,
        "system": "MemorySystem (our implementation)"
    }

    total_start = time.time()
    ground_truth = dataset.get_memories()

    # =========================================================================
    # PHASE 1: EXTRACTION
    # Uses: memory_system.extract_memories_from_dialogue()
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: EXTRACTION (using MemorySystem)")
    logger.info("=" * 60)

    start_time = time.time()
    dialogues = dataset.get_dialogues()

    extraction_results = {}
    total_extracted = 0

    for i, dialogue in enumerate(dialogues):
        # Format dialogue as text
        dialogue_text = "\n".join([
            f"{turn.role}: {turn.content}"
            for turn in dialogue
        ])

        logger.info(f"Extracting from dialogue {i+1}/{len(dialogues)}")

        try:
            # Use our MemorySystem's extraction method
            memories = await memory_system.extract_memories_from_dialogue(
                dialogue=dialogue_text,
                primary_model_fn=gpt5_1,
                auto_store=True  # Automatically adds to storage
            )
            extraction_results[f"dialogue_{i}"] = memories
            total_extracted += len(memories)

        except Exception as e:
            logger.error(f"Error extracting dialogue {i}: {e}")
            extraction_results[f"dialogue_{i}"] = []

        # Small delay to avoid rate limits
        if i % 5 == 4:
            await asyncio.sleep(1)

    # Generate embeddings for prefiltering
    if enable_prefilter:
        # Update top_k: max(1000, 10% of memories)
        mem_count = memory_system.count_memories()
        top_k = max(1000, int(mem_count * 0.1))
        memory_system.update_prefilter_config(top_k=top_k)
        logger.info(f"Prefilter top_k set to {top_k} (max of 1000 or 10% of {mem_count})")

        logger.info("Generating embeddings for prefiltering...")
        await memory_system.generate_embeddings()

    # Save extraction results
    with open(run_dir / "extraction_raw.json", 'w') as f:
        json.dump(extraction_results, f, indent=2)

    # Compute extraction metrics
    all_extracted = []
    for memories in extraction_results.values():
        all_extracted.extend(memories)

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

    mem_stats = memory_system.get_stats()
    results["extraction"] = {
        "metrics": extraction_metrics.to_dict(),
        "time_seconds": time.time() - start_time,
        "total_extracted": total_extracted,
        "matched": len(matched_gt_ids),
        "memories_in_store": mem_stats["total_memories"]
    }

    logger.info(f"Extraction complete: Recall={extraction_metrics.recall:.2%}, "
                f"Precision={extraction_metrics.precision:.2%}, "
                f"Memories in store: {mem_stats['total_memories']}")

    # =========================================================================
    # PHASE 2: UPDATING
    # Uses: memory_system.process_new_information()
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: UPDATING (using MemorySystem)")
    logger.info("=" * 60)

    start_time = time.time()
    scenarios = dataset.get_update_scenarios()

    update_results = {}
    predictions = []
    ground_truth_actions = []
    action_types = []

    for i, scenario in enumerate(scenarios):
        logger.info(f"Processing update {i+1}/{len(scenarios)}")

        try:
            # Use our MemorySystem's update processing
            result = await memory_system.process_new_information(
                new_info=scenario.new_information,
                primary_model_fn=gpt5_1,
                similarity_threshold=0.5  # Match relevance_threshold
            )

            # Determine what action was taken
            actions_taken = result.get("actions_taken", [])
            if not actions_taken:
                predicted_action = "NOOP"
            else:
                # Use the first action (could have multiple)
                predicted_action = actions_taken[0].get("action", "NOOP")

            update_results[scenario.scenario_id] = {
                "predicted": predicted_action,
                "expected": scenario.expected_action,
                "actions_taken": actions_taken
            }
            predictions.append(predicted_action)

        except Exception as e:
            logger.error(f"Error processing update {scenario.scenario_id}: {e}")
            predictions.append("ERROR")
            update_results[scenario.scenario_id] = {"error": str(e)}

        ground_truth_actions.append(scenario.expected_action)
        action_types.append(scenario.conflict_type)

        # Small delay
        if i % 10 == 9:
            await asyncio.sleep(1)

    # Save update results
    with open(run_dir / "updating_raw.json", 'w') as f:
        json.dump(update_results, f, indent=2)

    update_metrics = compute_update_metrics(
        predictions=predictions,
        ground_truth=ground_truth_actions,
        action_types=action_types
    )

    mem_stats = memory_system.get_stats()
    results["updating"] = {
        "metrics": update_metrics.to_dict(),
        "time_seconds": time.time() - start_time,
        "total_scenarios": len(scenarios),
        "final_memory_count": mem_stats["total_memories"]
    }

    logger.info(f"Updating complete: Accuracy={update_metrics.accuracy:.2%}, "
                f"Final memories: {mem_stats['total_memories']}")

    # =========================================================================
    # PHASE 3: QA
    # Uses: memory_system.retrieve_relevant_memories() + query()
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: QA (using MemorySystem)")
    logger.info("=" * 60)

    start_time = time.time()
    qa_pairs = dataset.get_qa_pairs(limit=sample_size)

    qa_results = {}
    correctness_scores = []
    hallucination_flags = []
    omission_flags = []
    question_types = []

    for i, qa in enumerate(qa_pairs):
        logger.info(f"Answering question {i+1}/{len(qa_pairs)}")

        try:
            # Use our MemorySystem's query method
            answer = await memory_system.query(
                context=qa.question,
                task=f"Answer this question based on what you know about the user: {qa.question}",
                primary_model_fn=gpt5_1,
                include_scores=True
            )

            # Judge the answer
            judge_prompt = f"""You are evaluating an AI's answer to a question based on stored memories.

QUESTION: {qa.question}

GROUND TRUTH ANSWER: {qa.ground_truth_answer}

AI'S ANSWER: {answer}

Evaluate:
1. Correctness (0.0 to 1.0): Is it factually correct?
2. Hallucination: Does it contain made-up information?
3. Omission: Does it miss important information?

Respond in exactly this format:
CORRECTNESS: <0.0 to 1.0>
HALLUCINATION: <YES|NO>
OMISSION: <YES|NO>
REASONING: <brief explanation>"""

            judgment = await gpt5_1(judge_prompt)

            # Parse judgment
            correctness = 0.0
            hallucination = False
            omission = False
            reasoning = ""

            for line in judgment.strip().split('\n'):
                if line.startswith("CORRECTNESS:"):
                    try:
                        correctness = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("HALLUCINATION:"):
                    hallucination = "YES" in line.upper()
                elif line.startswith("OMISSION:"):
                    omission = "YES" in line.upper()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            qa_results[qa.qa_id] = {
                "question": qa.question,
                "answer": answer,
                "ground_truth": qa.ground_truth_answer,
                "correctness": correctness,
                "hallucination": hallucination,
                "omission": omission,
                "reasoning": reasoning
            }

            correctness_scores.append(correctness)
            hallucination_flags.append(hallucination)
            omission_flags.append(omission)
            question_types.append(qa.question_type)

        except Exception as e:
            logger.error(f"Error answering {qa.qa_id}: {e}")
            correctness_scores.append(0.0)
            hallucination_flags.append(False)
            omission_flags.append(True)
            question_types.append(qa.question_type)
            qa_results[qa.qa_id] = {"error": str(e)}

        # Small delay
        if i % 5 == 4:
            await asyncio.sleep(1)

    # Save QA results
    with open(run_dir / "qa_raw.json", 'w') as f:
        json.dump(qa_results, f, indent=2)

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

    logger.info(f"QA complete: Correctness={qa_metrics.correctness:.2%}, "
                f"Hallucination={qa_metrics.hallucination_rate:.2%}")

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    results["total_time_seconds"] = time.time() - total_start
    results["completed_at"] = datetime.now().isoformat()
    results["final_stats"] = memory_system.get_stats()

    with open(run_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("HALUMEM BENCHMARK - MEMORYSYSTEM RESULTS")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"System: MemorySystem (our implementation)")
    print(f"Variant: {variant}")
    print(f"Prefilter: {'enabled' if enable_prefilter else 'disabled'}")
    print(f"Output: {run_dir}")
    print()

    print("EXTRACTION:")
    m = results["extraction"]["metrics"]
    print(f"  Recall: {m.get('recall', 0):.2%}")
    print(f"  Weighted Recall: {m.get('weighted_recall', 0):.2%}")
    print(f"  Precision: {m.get('precision', 0):.2%}")
    print(f"  Memories in store: {results['extraction']['memories_in_store']}")
    print()

    print("UPDATING:")
    m = results["updating"]["metrics"]
    print(f"  Accuracy: {m.get('accuracy', 0):.2%}")
    print(f"  Omission Rate: {m.get('omission_rate', 0):.2%}")
    print(f"  Hallucination Rate: {m.get('hallucination_rate', 0):.2%}")
    print()

    print("QA:")
    m = results["qa"]["metrics"]
    print(f"  Correctness: {m.get('correctness', 0):.2%}")
    print(f"  Hallucination Rate: {m.get('hallucination_rate', 0):.2%}")
    print(f"  Omission Rate: {m.get('omission_rate', 0):.2%}")
    print()

    print(f"Total time: {results['total_time_seconds']:.1f}s")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run HaluMem benchmark using our MemorySystem"
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
        "--no-prefilter",
        action="store_true",
        help="Disable embedding prefiltering"
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

    asyncio.run(run_memory_system_benchmark(
        variant=args.variant,
        sample_size=args.sample_size,
        enable_prefilter=not args.no_prefilter
    ))


if __name__ == "__main__":
    main()
