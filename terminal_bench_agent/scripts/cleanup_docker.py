#!/usr/bin/env python3
"""Script to cleanup Docker containers used in Terminal-Bench testing.

This script provides cleanup for Docker containers with retry logic
and proper error handling.

Usage:
    # List all containers
    python scripts/cleanup_docker.py --list

    # Cleanup exited containers
    python scripts/cleanup_docker.py --cleanup-exited

    # Cleanup containers older than 24 hours
    python scripts/cleanup_docker.py --cleanup-old --hours 24

    # Cleanup all Terminal-Bench containers
    python scripts/cleanup_docker.py --cleanup-all --label terminal-bench
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from cleanup_manager import DockerCleanupManager, CleanupError
    CLEANUP_MANAGER_AVAILABLE = True
except ImportError:
    CLEANUP_MANAGER_AVAILABLE = False
    print("Error: cleanup_manager not available")
    sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cleanup Docker containers used in Terminal-Bench testing"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all Docker containers"
    )
    parser.add_argument(
        "--list-running",
        action="store_true",
        help="List only running containers"
    )
    parser.add_argument(
        "--cleanup-exited",
        action="store_true",
        help="Cleanup exited containers"
    )
    parser.add_argument(
        "--cleanup-old",
        action="store_true",
        help="Cleanup old containers"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Age threshold in hours for cleanup (default: 24)"
    )
    parser.add_argument(
        "--cleanup-all",
        action="store_true",
        help="Cleanup all containers matching filters"
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Filter by label (e.g., 'terminal-bench' or 'harbor')"
    )
    parser.add_argument(
        "--status",
        type=str,
        help="Filter by status (e.g., 'exited', 'created', 'dead')"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first cleanup failure"
    )
    parser.add_argument(
        "--stop-timeout",
        type=int,
        default=10,
        help="Timeout in seconds for graceful container stop (default: 10)"
    )

    args = parser.parse_args()

    if not CLEANUP_MANAGER_AVAILABLE:
        print("Error: cleanup_manager module not available")
        sys.exit(1)

    # Initialize Docker cleanup manager
    try:
        manager = DockerCleanupManager()
    except CleanupError as e:
        print(f"Error initializing Docker cleanup manager: {e}")
        sys.exit(1)

    # List containers
    if args.list or args.list_running:
        filters = {}
        if args.label:
            filters["label"] = args.label
        if args.status:
            filters["status"] = args.status

        all_containers = not args.list_running
        containers = manager.list_containers(all_containers=all_containers, filters=filters)

        if not containers:
            print("No containers found.")
        else:
            print(f"\nFound {len(containers)} container(s):\n")
            for cid in containers:
                print(f"  - {cid}")
            print()

        print("\nTo cleanup containers:")
        print("  - Cleanup exited: python cleanup_docker.py --cleanup-exited")
        print("  - Cleanup old (24h): python cleanup_docker.py --cleanup-old --hours 24")
        print("  - Cleanup by label: python cleanup_docker.py --cleanup-all --label terminal-bench")
        print()

    # Cleanup exited containers
    elif args.cleanup_exited:
        print("Cleaning up exited containers...\n")
        filters = {"status": "exited"}
        if args.label:
            filters["label"] = args.label

        try:
            deleted, failed = manager.cleanup_containers(
                filters=filters,
                stop_timeout=args.stop_timeout,
                fail_fast=args.fail_fast
            )
            print(f"\n✓ Cleaned up {deleted} exited container(s)")
            if failed > 0:
                print(f"⚠  {failed} cleanup(s) failed (after retries)")
        except CleanupError as e:
            print(f"\n✗ Cleanup failed: {e}")
            sys.exit(1)

    # Cleanup old containers
    elif args.cleanup_old:
        print(f"Cleaning up containers older than {args.hours} hour(s)...\n")
        try:
            deleted, failed = manager.cleanup_old_containers(
                max_age_hours=args.hours,
                fail_fast=args.fail_fast
            )
            print(f"\n✓ Cleaned up {deleted} old container(s)")
            if failed > 0:
                print(f"⚠  {failed} cleanup(s) failed (after retries)")
        except CleanupError as e:
            print(f"\n✗ Cleanup failed: {e}")
            sys.exit(1)

    # Cleanup all matching containers
    elif args.cleanup_all:
        filters = {}
        if args.label:
            filters["label"] = args.label
        if args.status:
            filters["status"] = args.status

        if not filters:
            print("Error: --cleanup-all requires at least one filter (--label or --status)")
            print("To prevent accidental deletion of all containers")
            sys.exit(1)

        print(f"Cleaning up containers matching filters: {filters}\n")
        try:
            deleted, failed = manager.cleanup_containers(
                filters=filters,
                stop_timeout=args.stop_timeout,
                fail_fast=args.fail_fast
            )
            print(f"\n✓ Cleaned up {deleted} container(s)")
            if failed > 0:
                print(f"⚠  {failed} cleanup(s) failed (after retries)")
        except CleanupError as e:
            print(f"\n✗ Cleanup failed: {e}")
            sys.exit(1)

    else:
        # Default: list all containers
        containers = manager.list_containers(all_containers=True)
        if not containers:
            print("No containers found.")
        else:
            print(f"\nFound {len(containers)} container(s):\n")
            for cid in containers:
                print(f"  - {cid}")
            print()

        print("\nTo cleanup containers:")
        print("  - Cleanup exited: python cleanup_docker.py --cleanup-exited")
        print("  - Cleanup old (24h): python cleanup_docker.py --cleanup-old --hours 24")
        print("  - Cleanup by label: python cleanup_docker.py --cleanup-all --label terminal-bench")
        print()

    print("✓ Done!")


if __name__ == "__main__":
    main()
