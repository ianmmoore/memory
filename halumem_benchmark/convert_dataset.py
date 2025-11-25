#!/usr/bin/env python3
"""Convert HaluMem raw JSONL files to benchmark format.

This script converts the raw HaluMem dataset files from HuggingFace
to the format expected by our benchmark runner.

Usage:
    python convert_dataset.py --variant long
    python convert_dataset.py --variant medium
    python convert_dataset.py --variant both
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
RAW_DIR = Path(__file__).parent.parent / "data" / "halumem" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "halumem"


def convert_variant(variant: str, raw_dir: Path, output_dir: Path) -> Dict[str, int]:
    """Convert a single variant (medium or long) to benchmark format.

    Args:
        variant: Either "medium" or "long".
        raw_dir: Directory containing raw JSONL files.
        output_dir: Base output directory.

    Returns:
        Statistics about the converted data.
    """
    # Map variant to filename
    filename = f"HaluMem-{'Long' if variant == 'long' else 'Medium'}.jsonl"
    input_path = raw_dir / filename

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {input_path} to {variant_dir}")

    # Accumulators
    all_dialogues = []
    all_memories = []
    all_qa_pairs = []
    all_update_scenarios = []
    importance_weights = {}

    # Memory ID tracking for linking QA to memories
    memory_content_to_id = {}

    # Read and process each user
    with open(input_path, 'r', encoding='utf-8') as f:
        for user_idx, line in enumerate(f):
            if not line.strip():
                continue

            user_data = json.loads(line)
            user_uuid = user_data.get('uuid', f'user_{user_idx}')

            logger.info(f"Processing user {user_idx + 1}: {user_uuid[:8]}... ({user_data.get('total_question_count', 0)} questions)")

            sessions = user_data.get('sessions', [])

            for session_idx, session in enumerate(sessions):
                session_id = session.get('session_id', session_idx)

                # Process dialogue
                dialogue_turns = []
                for turn in session.get('dialogue', []):
                    dialogue_turns.append({
                        "role": turn.get('role', 'user'),
                        "content": turn.get('content', ''),
                        "turn_id": turn.get('dialogue_turn', len(dialogue_turns)),
                        "timestamp": turn.get('timestamp', '')
                    })

                if dialogue_turns:
                    all_dialogues.append({
                        "dialogue_id": f"{user_uuid}_s{session_id}",
                        "user_id": user_uuid,
                        "session_id": session_id,
                        "turns": dialogue_turns,
                        "token_length": session.get('dialogue_token_length', 0)
                    })

                # Process memory points
                for mem_idx, mem in enumerate(session.get('memory_points', [])):
                    memory_id = f"m_{user_idx}_{session_id}_{mem_idx}"
                    memory_content = mem.get('memory_content', '')

                    # Track for linking QA evidence
                    memory_content_to_id[memory_content.lower().strip()] = memory_id

                    memory_entry = {
                        "memory_id": memory_id,
                        "text": memory_content,
                        "source_turns": [mem.get('event_source', 0)] if mem.get('event_source') else [],
                        "importance": mem.get('importance', 1.0),
                        "category": mem.get('memory_type', 'unknown'),
                        "memory_source": mem.get('memory_source', 'unknown'),
                        "is_target": mem.get('memory_source') == 'primary',
                        "is_update": mem.get('is_update') == 'True' or mem.get('is_update') is True,
                        "timestamp": mem.get('timestamp', ''),
                        "user_id": user_uuid,
                        "session_id": session_id
                    }
                    all_memories.append(memory_entry)
                    importance_weights[memory_id] = mem.get('importance', 1.0)

                    # Create update scenarios from memories marked as updates
                    if memory_entry['is_update']:
                        original_memories = mem.get('original_memories', [])
                        for orig_idx, orig in enumerate(original_memories):
                            orig_text = orig if isinstance(orig, str) else orig.get('memory_content', '')
                            if orig_text:
                                all_update_scenarios.append({
                                    "scenario_id": f"u_{user_idx}_{session_id}_{mem_idx}_{orig_idx}",
                                    "old_memory": orig_text,
                                    "new_information": memory_content,
                                    "expected_action": "UPDATE",
                                    "expected_result": memory_content,
                                    "conflict_type": "refinement",
                                    "user_id": user_uuid
                                })

                # Process questions
                for q_idx, q in enumerate(session.get('questions', [])):
                    # Try to link evidence to memory IDs
                    evidence = q.get('evidence', [])
                    relevant_memory_ids = []
                    for ev in evidence:
                        ev_text = ev if isinstance(ev, str) else ev.get('memory_content', '')
                        ev_key = ev_text.lower().strip()
                        if ev_key in memory_content_to_id:
                            relevant_memory_ids.append(memory_content_to_id[ev_key])

                    all_qa_pairs.append({
                        "qa_id": f"q_{user_idx}_{session_id}_{q_idx}",
                        "question": q.get('question', ''),
                        "ground_truth_answer": q.get('answer', ''),
                        "relevant_memory_ids": relevant_memory_ids,
                        "evidence": evidence,
                        "question_type": q.get('question_type', 'unknown'),
                        "difficulty": q.get('difficulty', 'medium'),
                        "user_id": user_uuid,
                        "session_id": session_id
                    })

    # Save all files
    logger.info(f"Saving {len(all_dialogues)} dialogues...")
    with open(variant_dir / "dialogues.jsonl", 'w', encoding='utf-8') as f:
        for d in all_dialogues:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Saving {len(all_memories)} memories...")
    with open(variant_dir / "ground_truth_memories.jsonl", 'w', encoding='utf-8') as f:
        for m in all_memories:
            json.dump(m, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Saving {len(all_qa_pairs)} QA pairs...")
    with open(variant_dir / "qa_pairs.jsonl", 'w', encoding='utf-8') as f:
        for qa in all_qa_pairs:
            json.dump(qa, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Saving {len(all_update_scenarios)} update scenarios...")
    with open(variant_dir / "update_scenarios.jsonl", 'w', encoding='utf-8') as f:
        for u in all_update_scenarios:
            json.dump(u, f, ensure_ascii=False)
            f.write('\n')

    logger.info("Saving importance weights...")
    with open(variant_dir / "importance_weights.json", 'w', encoding='utf-8') as f:
        json.dump(importance_weights, f, indent=2)

    # Compute statistics
    stats = {
        "dialogues": len(all_dialogues),
        "memories": len(all_memories),
        "qa_pairs": len(all_qa_pairs),
        "update_scenarios": len(all_update_scenarios),
        "total_turns": sum(len(d['turns']) for d in all_dialogues),
        "total_tokens": sum(d.get('token_length', 0) for d in all_dialogues)
    }

    # Count by question type
    question_types = defaultdict(int)
    for qa in all_qa_pairs:
        question_types[qa['question_type']] += 1

    # Count by memory source
    memory_sources = defaultdict(int)
    for m in all_memories:
        memory_sources[m['memory_source']] += 1

    # Create README
    readme_content = f"""# HaluMem-{variant.capitalize()} Dataset

Downloaded from: https://huggingface.co/datasets/IAAR-Shanghai/HaluMem
Paper: arXiv:2511.03506

## Statistics
- Dialogues: {stats['dialogues']:,}
- Total dialogue turns: {stats['total_turns']:,}
- Total tokens: {stats['total_tokens']:,}
- Memory points: {stats['memories']:,}
- QA pairs: {stats['qa_pairs']:,}
- Update scenarios: {stats['update_scenarios']:,}

## Question Types
{chr(10).join(f'- {qt}: {count}' for qt, count in sorted(question_types.items()))}

## Memory Sources
{chr(10).join(f'- {ms}: {count}' for ms, count in sorted(memory_sources.items()))}

## Files
- dialogues.jsonl: Conversation turns
- ground_truth_memories.jsonl: Expected extracted memories
- qa_pairs.jsonl: Question-answer pairs for evaluation
- update_scenarios.jsonl: Memory update test cases
- importance_weights.json: Weights for weighted recall
"""
    with open(variant_dir / "README.md", 'w') as f:
        f.write(readme_content)

    logger.info(f"Conversion complete for HaluMem-{variant.capitalize()}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert HaluMem dataset to benchmark format"
    )
    parser.add_argument(
        "--variant",
        choices=["medium", "long", "both"],
        default="long",
        help="Which variant to convert (default: long)"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory with raw JSONL files (default: {RAW_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    args = parser.parse_args()

    variants = ["medium", "long"] if args.variant == "both" else [args.variant]

    for variant in variants:
        try:
            stats = convert_variant(variant, args.raw_dir, args.output_dir)
            print(f"\n✅ HaluMem-{variant.capitalize()} converted successfully!")
            print(f"   Location: {args.output_dir / variant}")
            print(f"   Dialogues: {stats['dialogues']:,}")
            print(f"   Memories: {stats['memories']:,}")
            print(f"   QA pairs: {stats['qa_pairs']:,}")
            print(f"   Update scenarios: {stats['update_scenarios']:,}")
            print(f"   Total tokens: {stats['total_tokens']:,}")
        except FileNotFoundError as e:
            logger.error(str(e))
            print(f"\n❌ Error: {e}")
            print("   Run download_dataset.py first to download the raw files.")


if __name__ == "__main__":
    main()
