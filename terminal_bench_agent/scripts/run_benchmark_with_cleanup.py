#!/usr/bin/env python3
"""Script to run Terminal-Bench with automatic Daytona cleanup.

This wrapper automatically cleans up old sandboxes before running
the benchmark to avoid hitting disk limits.

Usage:
    python scripts/run_benchmark_with_cleanup.py
    python scripts/run_benchmark_with_cleanup.py --skip-cleanup
    python scripts/run_benchmark_with_cleanup.py --cleanup-days 2
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from daytona import Daytona
    DAYTONA_AVAILABLE = True
except ImportError:
    DAYTONA_AVAILABLE = False
    print("Warning: Daytona SDK not installed. Skipping cleanup.")


def cleanup_sandboxes(max_age_days: int = 1, also_delete_stopped: bool = True):
    """Cleanup old Daytona sandboxes before running benchmark.

    Args:
        max_age_days: Delete sandboxes older than this many days
        also_delete_stopped: Also delete stopped/failed sandboxes regardless of age
    """
    if not DAYTONA_AVAILABLE:
        print("⚠️  Daytona SDK not available, skipping cleanup")
        return

    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        print("⚠️  DAYTONA_API_KEY not set, skipping cleanup")
        return

    print("\n" + "="*80)
    print("CLEANING UP OLD DAYTONA SANDBOXES")
    print("="*80 + "\n")

    try:
        # Daytona reads from DAYTONA_API_KEY environment variable automatically
        client = Daytona()

        # List current sandboxes
        print("Fetching sandbox list...")
        result = client.list()
        sandboxes = getattr(result, 'items', [])
        print(f"Found {len(sandboxes)} sandbox(es)\n")

        if len(sandboxes) == 0:
            print("No sandboxes to clean up.\n")
            return

        deleted_count = 0

        # Delete stopped/failed sandboxes first
        if also_delete_stopped:
            print("Deleting stopped/failed sandboxes...")
            for sandbox in sandboxes:
                try:
                    status = getattr(sandbox, 'status', '').lower()
                    sandbox_id = getattr(sandbox, 'id', 'unknown')

                    if status in ['stopped', 'terminated', 'failed', 'error', 'exited']:
                        print(f"  Deleting {sandbox_id} (status: {status})...")
                        try:
                            client.delete(sandbox_id)
                            print(f"    ✓ Deleted")
                            deleted_count += 1
                        except Exception as e:
                            print(f"    ✗ Failed: {e}")
                except Exception as e:
                    print(f"  Error processing sandbox: {e}")

        # Delete old sandboxes
        cutoff_date = datetime.now(datetime.now().astimezone().tzinfo) - timedelta(days=max_age_days)

        print(f"\nDeleting sandboxes older than {max_age_days} day(s)...")
        for sandbox in sandboxes:
            try:
                created = getattr(sandbox, 'created_at', None)
                if not created:
                    continue

                # Parse created timestamp
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                elif isinstance(created, datetime):
                    created_dt = created
                else:
                    continue

                sandbox_id = getattr(sandbox, 'id', 'unknown')
                status = getattr(sandbox, 'status', '').lower()

                # Skip if already deleted
                if status in ['stopped', 'terminated', 'failed', 'error', 'exited']:
                    continue

                # Check if older than cutoff
                if created_dt < cutoff_date:
                    print(f"  Deleting {sandbox_id} (created {created_dt.strftime('%Y-%m-%d %H:%M')})...")
                    try:
                        client.delete(sandbox_id)
                        print(f"    ✓ Deleted")
                        deleted_count += 1
                    except Exception as e:
                        print(f"    ✗ Failed: {e}")

            except Exception as e:
                print(f"  Error processing sandbox: {e}")

        print(f"\n✓ Cleanup complete! Removed {deleted_count} sandbox(es)\n")

    except Exception as e:
        print(f"⚠️  Cleanup failed: {e}")
        print("Continuing with benchmark anyway...\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Terminal-Bench with automatic Daytona cleanup"
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip automatic sandbox cleanup"
    )
    parser.add_argument(
        "--cleanup-days",
        type=int,
        default=1,
        help="Delete sandboxes older than this many days (default: 1)"
    )
    parser.add_argument(
        "--dataset",
        default="terminus-2",
        help="Dataset name (default: terminus-2)"
    )
    parser.add_argument(
        "--version",
        default="terminal-bench",
        help="Dataset version (default: terminal-bench)"
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Specific task IDs to run"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--model",
        help="Model name (uses OPENAI_MODEL env var if not specified)"
    )

    args = parser.parse_args()

    # Perform cleanup unless skipped
    if not args.skip_cleanup:
        cleanup_sandboxes(
            max_age_days=args.cleanup_days,
            also_delete_stopped=True
        )
    else:
        print("\n⚠️  Skipping cleanup (--skip-cleanup specified)\n")

    # Build benchmark command
    import subprocess

    cmd = [
        "harbor", "run",
        "--agent-import-path", "terminal_bench_agent.core:MemoryGuidedAgent",
        "--dataset", args.dataset,
        "--env", "daytona",  # Use Daytona environment
    ]

    if args.tasks:
        for task_id in args.tasks:
            cmd.extend(["--task-name", task_id])

    if args.output_dir:
        cmd.extend(["--jobs-dir", args.output_dir])
    else:
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        output_dir = f"./terminal_bench_agent/jobs/{timestamp}"
        cmd.extend(["--jobs-dir", output_dir])

    if args.parallel > 1:
        cmd.extend(["--n-concurrent", str(args.parallel)])

    # Add model if specified
    env = os.environ.copy()
    if args.model:
        env["OPENAI_MODEL"] = args.model
        cmd.extend(["--model", args.model])

    print("="*80)
    print("RUNNING HARBOR WITH DAYTONA")
    print("="*80)
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Output: {output_dir if not args.output_dir else args.output_dir}\n")

    # Run Terminal-Bench
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print(f"\n{'='*80}")
        print("✓ BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}\n")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"✗ BENCHMARK FAILED (exit code {e.returncode})")
        print(f"{'='*80}\n")
        return e.returncode
    except FileNotFoundError:
        print("\n✗ Error: 'tb' command not found.")
        print("Please install Terminal-Bench: pip install terminal-bench")
        return 1


if __name__ == "__main__":
    sys.exit(main())
