#!/usr/bin/env python3
"""Download and setup the HaluMem dataset from HuggingFace.

This script downloads the official HaluMem dataset (arXiv:2511.03506) and
converts it to the format expected by our benchmark runner.

Usage:
    python download_dataset.py --variant long
    python download_dataset.py --variant medium
    python download_dataset.py --variant both
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "halumem"


def download_from_huggingface(variant: str = "long") -> Any:
    """Download the HaluMem dataset from HuggingFace.

    Args:
        variant: Either "medium" or "long".

    Returns:
        The loaded dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install the 'datasets' library: pip install datasets")
        sys.exit(1)

    logger.info(f"Downloading HaluMem-{variant} from HuggingFace...")

    # The dataset name on HuggingFace
    dataset_name = "IAAR-Shanghai/HaluMem"

    # Load the specific variant
    # HaluMem has configurations for 'medium' and 'long'
    config_name = f"Halu-{'Medium' if variant == 'medium' else 'Long'}"

    try:
        dataset = load_dataset(dataset_name, config_name, trust_remote_code=True)
        logger.info(f"Successfully downloaded {config_name}")
        return dataset
    except Exception as e:
        logger.warning(f"Could not load with config '{config_name}': {e}")
        # Try loading without config name
        try:
            dataset = load_dataset(dataset_name, trust_remote_code=True)
            logger.info("Loaded dataset without specific config")
            return dataset
        except Exception as e2:
            logger.error(f"Failed to download dataset: {e2}")
            raise


def convert_to_benchmark_format(
    dataset: Any,
    output_dir: Path,
    variant: str
) -> None:
    """Convert HuggingFace dataset to our benchmark format.

    Args:
        dataset: The HuggingFace dataset.
        output_dir: Directory to save converted files.
        variant: "medium" or "long".
    """
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting dataset to benchmark format in {variant_dir}")

    # Get the data (could be in 'train' split or directly accessible)
    if hasattr(dataset, 'keys'):
        # It's a DatasetDict
        if 'train' in dataset:
            data = dataset['train']
        else:
            # Use the first available split
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
    else:
        data = dataset

    # Extract and save each component
    dialogues = []
    memories = []
    qa_pairs = []
    update_scenarios = []
    importance_weights = {}

    for idx, user_data in enumerate(data):
        logger.info(f"Processing user {idx + 1}/{len(data)}")

        # Process sessions/dialogues
        sessions = user_data.get('sessions', [])
        if isinstance(sessions, str):
            sessions = json.loads(sessions)

        for session_idx, session in enumerate(sessions):
            # Extract dialogue turns
            dialogue_turns = []
            dialogue = session.get('dialogue', [])
            if isinstance(dialogue, str):
                dialogue = json.loads(dialogue)

            for turn in dialogue:
                dialogue_turns.append({
                    "role": turn.get('role', 'user'),
                    "content": turn.get('content', ''),
                    "turn_id": turn.get('dialogue_turn', len(dialogue_turns)),
                    "timestamp": turn.get('timestamp', '')
                })

            if dialogue_turns:
                dialogues.append({
                    "dialogue_id": f"user{idx}_session{session_idx}",
                    "user_id": user_data.get('uuid', f'user_{idx}'),
                    "turns": dialogue_turns
                })

            # Extract memory points
            memory_points = session.get('memory_points', [])
            if isinstance(memory_points, str):
                memory_points = json.loads(memory_points)

            for mem_idx, mem in enumerate(memory_points):
                memory_id = f"m_{idx}_{session_idx}_{mem_idx}"
                memories.append({
                    "memory_id": memory_id,
                    "text": mem.get('memory_content', ''),
                    "source_turns": [],  # Would need mapping from original
                    "importance": mem.get('importance', 1.0),
                    "category": mem.get('memory_type', 'unknown'),
                    "is_target": mem.get('memory_source') == 'primary',
                    "is_update": mem.get('is_update', False),
                    "original_memories": mem.get('original_memories', [])
                })
                importance_weights[memory_id] = mem.get('importance', 1.0)

                # Create update scenarios from updated memories
                if mem.get('is_update') and mem.get('original_memories'):
                    for orig in mem.get('original_memories', []):
                        update_scenarios.append({
                            "scenario_id": f"u_{idx}_{session_idx}_{mem_idx}",
                            "old_memory": orig if isinstance(orig, str) else orig.get('memory_content', ''),
                            "new_information": mem.get('memory_content', ''),
                            "expected_action": "UPDATE",
                            "expected_result": mem.get('memory_content', ''),
                            "conflict_type": "refinement"
                        })

            # Extract QA pairs
            questions = session.get('questions', [])
            if isinstance(questions, str):
                questions = json.loads(questions)

            for q_idx, q in enumerate(questions):
                qa_pairs.append({
                    "qa_id": f"q_{idx}_{session_idx}_{q_idx}",
                    "question": q.get('question', q.get('Question', '')),
                    "ground_truth_answer": q.get('answer', q.get('Answer', '')),
                    "relevant_memory_ids": [],  # Would need to map from evidence
                    "question_type": q.get('question_type', q.get('QuestionType', 'single-hop')),
                    "difficulty": q.get('difficulty', q.get('Difficulty', 'medium'))
                })

    # Save to files
    logger.info(f"Saving {len(dialogues)} dialogues...")
    with open(variant_dir / "dialogues.jsonl", 'w') as f:
        for d in dialogues:
            json.dump(d, f)
            f.write('\n')

    logger.info(f"Saving {len(memories)} memories...")
    with open(variant_dir / "ground_truth_memories.jsonl", 'w') as f:
        for m in memories:
            json.dump(m, f)
            f.write('\n')

    logger.info(f"Saving {len(qa_pairs)} QA pairs...")
    with open(variant_dir / "qa_pairs.jsonl", 'w') as f:
        for qa in qa_pairs:
            json.dump(qa, f)
            f.write('\n')

    logger.info(f"Saving {len(update_scenarios)} update scenarios...")
    with open(variant_dir / "update_scenarios.jsonl", 'w') as f:
        for u in update_scenarios:
            json.dump(u, f)
            f.write('\n')

    logger.info("Saving importance weights...")
    with open(variant_dir / "importance_weights.json", 'w') as f:
        json.dump(importance_weights, f, indent=2)

    # Create README
    readme_content = f"""# HaluMem-{variant.capitalize()} Dataset

Downloaded from: https://huggingface.co/datasets/IAAR-Shanghai/HaluMem
Paper: arXiv:2511.03506

## Statistics
- Dialogues: {len(dialogues)}
- Memory points: {len(memories)}
- QA pairs: {len(qa_pairs)}
- Update scenarios: {len(update_scenarios)}

## Files
- dialogues.jsonl: Conversation turns
- ground_truth_memories.jsonl: Expected extracted memories
- qa_pairs.jsonl: Question-answer pairs for evaluation
- update_scenarios.jsonl: Memory update test cases
- importance_weights.json: Weights for weighted recall
"""
    with open(variant_dir / "README.md", 'w') as f:
        f.write(readme_content)

    logger.info(f"Dataset saved to {variant_dir}")
    print(f"\n✅ HaluMem-{variant.capitalize()} dataset ready!")
    print(f"   Location: {variant_dir}")
    print(f"   Dialogues: {len(dialogues)}")
    print(f"   Memories: {len(memories)}")
    print(f"   QA pairs: {len(qa_pairs)}")
    print(f"   Update scenarios: {len(update_scenarios)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download HaluMem dataset from HuggingFace"
    )
    parser.add_argument(
        "--variant",
        choices=["medium", "long", "both"],
        default="long",
        help="Which variant to download (default: long)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    variants = ["medium", "long"] if args.variant == "both" else [args.variant]

    for variant in variants:
        try:
            dataset = download_from_huggingface(variant)
            convert_to_benchmark_format(dataset, args.output_dir, variant)
        except Exception as e:
            logger.error(f"Failed to process {variant}: {e}")
            if len(variants) == 1:
                sys.exit(1)


if __name__ == "__main__":
    main()
