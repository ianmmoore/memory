#!/usr/bin/env python3
"""Run the HaluMem benchmark.

Usage:
    python run.py                          # Run full HaluMem-Long with prefiltering
    python run.py --variant medium         # Run HaluMem-Medium
    python run.py --no-prefilter           # Run without prefiltering (~$3,000)
    python run.py --sample-size 100        # Run on 100 QA pairs for testing
    python run.py --dry-run                # Validate setup without running
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from halumem_benchmark import (
    HaluMemConfig,
    HaluMemDataset,
    HaluMemBenchmark,
    run_benchmark
)


async def dry_run(variant: str = "long") -> None:
    """Validate the benchmark setup without running."""
    print("=" * 60)
    print("HaluMem Benchmark - Dry Run")
    print("=" * 60)

    config = HaluMemConfig.for_long() if variant == "long" else HaluMemConfig.for_medium()
    print(f"\nConfiguration:")
    print(f"  Variant: {config.variant}")
    print(f"  Prefilter enabled: {config.prefilter_enabled}")
    print(f"  Prefilter top-k: {config.prefilter_top_k}")
    print(f"  Relevance threshold: {config.relevance_threshold}")

    print("\nLoading dataset...")
    dataset = HaluMemDataset(config)
    await dataset.load()

    stats = dataset.get_stats()
    print(f"\nDataset loaded:")
    print(f"  Dialogues: {stats['num_dialogues']:,}")
    print(f"  Memories: {stats['num_memories']:,}")
    print(f"  QA pairs: {stats['num_qa_pairs']:,}")
    print(f"  Update scenarios: {stats['num_update_scenarios']:,}")

    print("\nQuestion type distribution:")
    qa_pairs = dataset.get_qa_pairs()
    type_counts = {}
    for qa in qa_pairs:
        type_counts[qa.question_type] = type_counts.get(qa.question_type, 0) + 1
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")

    # Cost estimate
    print("\nEstimated cost (with prefiltering):")
    print(f"  Extraction: ~$2")
    print(f"  Updating: ~$1")
    print(f"  QA: ~$62")
    print(f"  Total: ~$65")

    print("\nEstimated cost (without prefiltering):")
    print(f"  Extraction: ~$2")
    print(f"  Updating: ~$1")
    print(f"  QA: ~$3,000")
    print(f"  Total: ~$3,003")

    print("\n" + "=" * 60)
    print("Dry run complete. Ready to run benchmark!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run the HaluMem benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                          # Full HaluMem-Long with prefiltering
    python run.py --variant medium         # HaluMem-Medium (faster)
    python run.py --sample-size 50         # Quick test with 50 QA pairs
    python run.py --no-prefilter           # Without prefiltering (expensive!)
    python run.py --dry-run                # Validate setup only
        """
    )
    parser.add_argument(
        "--variant",
        choices=["medium", "long"],
        default="long",
        help="Dataset variant (default: long)"
    )
    parser.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Disable prefiltering (WARNING: ~$3,000 cost)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit number of QA pairs for testing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running benchmark"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Check for API key
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    # Warn about cost without prefiltering
    if not args.dry_run and args.no_prefilter:
        print("\n" + "!" * 60)
        print("WARNING: Running without prefiltering!")
        print("Estimated cost: ~$3,000 for full HaluMem-Long")
        print("!" * 60)
        response = input("\nContinue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    # Run
    if args.dry_run:
        asyncio.run(dry_run(args.variant))
    else:
        results = asyncio.run(run_benchmark(
            variant=args.variant,
            prefilter=not args.no_prefilter,
            sample_size=args.sample_size
        ))
        results.print_summary()


if __name__ == "__main__":
    main()
