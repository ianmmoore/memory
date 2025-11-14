#!/usr/bin/env python3
"""Comprehensive cleanup script for both Daytona and Docker containers.

This script provides a unified interface to cleanup both Daytona sandboxes
and Docker containers used in Terminal-Bench testing.

Usage:
    # Full cleanup (both Daytona and Docker)
    python scripts/cleanup_all.py

    # Only Daytona cleanup
    python scripts/cleanup_all.py --daytona-only

    # Only Docker cleanup
    python scripts/cleanup_all.py --docker-only

    # Aggressive cleanup (stopped/exited + old)
    python scripts/cleanup_all.py --aggressive

    # List resources without cleanup
    python scripts/cleanup_all.py --list
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from cleanup_manager import (
        DaytonaCleanupManager,
        DockerCleanupManager,
        CleanupError,
        RetryableCleanupError
    )
    CLEANUP_MANAGER_AVAILABLE = True
except ImportError:
    CLEANUP_MANAGER_AVAILABLE = False
    print("Error: cleanup_manager not available")
    sys.exit(1)


def cleanup_daytona(
    max_age_days: int = 1,
    cleanup_stopped: bool = True,
    fail_fast: bool = False
):
    """Cleanup Daytona sandboxes.

    Args:
        max_age_days: Delete sandboxes older than this many days
        cleanup_stopped: Also delete stopped/failed sandboxes
        fail_fast: Stop on first failure

    Returns:
        Tuple of (total_deleted, total_failed)
    """
    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        print("⚠️  DAYTONA_API_KEY not set, skipping Daytona cleanup")
        return 0, 0

    print("\n" + "="*80)
    print("DAYTONA SANDBOX CLEANUP")
    print("="*80 + "\n")

    try:
        manager = DaytonaCleanupManager(api_key=api_key)

        # List current state
        sandboxes = manager.list_sandboxes()
        print(f"Found {len(sandboxes)} Daytona sandbox(es)\n")

        if len(sandboxes) == 0:
            print("No sandboxes to clean up.\n")
            return 0, 0

        total_deleted = 0
        total_failed = 0

        # Cleanup stopped sandboxes
        if cleanup_stopped:
            print("Cleaning up stopped/failed sandboxes...")
            deleted, failed = manager.cleanup_stopped_sandboxes(fail_fast=fail_fast)
            total_deleted += deleted
            total_failed += failed
            print(f"  ✓ Deleted {deleted} stopped sandbox(es)")
            if failed > 0:
                print(f"  ⚠  {failed} deletion(s) failed\n")
            print()

        # Cleanup old sandboxes
        print(f"Cleaning up sandboxes older than {max_age_days} day(s)...")
        deleted, failed = manager.cleanup_old_sandboxes(
            max_age_days=max_age_days,
            fail_fast=fail_fast
        )
        total_deleted += deleted
        total_failed += failed
        print(f"  ✓ Deleted {deleted} old sandbox(es)")
        if failed > 0:
            print(f"  ⚠  {failed} deletion(s) failed")

        print(f"\nDaytona cleanup summary: {total_deleted} deleted, {total_failed} failed\n")
        return total_deleted, total_failed

    except CleanupError as e:
        print(f"\n✗ Daytona cleanup failed: {e}\n")
        return 0, 1
    except Exception as e:
        print(f"\n⚠️  Unexpected Daytona cleanup error: {e}\n")
        return 0, 1


def cleanup_docker(
    max_age_hours: int = 24,
    cleanup_exited: bool = True,
    fail_fast: bool = False
):
    """Cleanup Docker containers.

    Args:
        max_age_hours: Delete containers older than this many hours
        cleanup_exited: Also delete exited containers
        fail_fast: Stop on first failure

    Returns:
        Tuple of (total_deleted, total_failed)
    """
    print("\n" + "="*80)
    print("DOCKER CONTAINER CLEANUP")
    print("="*80 + "\n")

    try:
        manager = DockerCleanupManager()

        # List current state
        containers = manager.list_containers(all_containers=True)
        print(f"Found {len(containers)} Docker container(s)\n")

        if len(containers) == 0:
            print("No containers to clean up.\n")
            return 0, 0

        total_deleted = 0
        total_failed = 0

        # Cleanup exited containers
        if cleanup_exited:
            print("Cleaning up exited containers...")
            deleted, failed = manager.cleanup_containers(
                filters={"status": "exited"},
                fail_fast=fail_fast
            )
            total_deleted += deleted
            total_failed += failed
            print(f"  ✓ Cleaned {deleted} exited container(s)")
            if failed > 0:
                print(f"  ⚠  {failed} cleanup(s) failed")
            print()

        # Cleanup old containers
        print(f"Cleaning up containers older than {max_age_hours} hour(s)...")
        deleted, failed = manager.cleanup_old_containers(
            max_age_hours=max_age_hours,
            fail_fast=fail_fast
        )
        total_deleted += deleted
        total_failed += failed
        print(f"  ✓ Cleaned {deleted} old container(s)")
        if failed > 0:
            print(f"  ⚠  {failed} cleanup(s) failed")

        print(f"\nDocker cleanup summary: {total_deleted} deleted, {total_failed} failed\n")
        return total_deleted, total_failed

    except CleanupError as e:
        print(f"\n✗ Docker cleanup failed: {e}\n")
        return 0, 1
    except Exception as e:
        print(f"\n⚠️  Unexpected Docker cleanup error: {e}\n")
        return 0, 1


def list_resources():
    """List all Daytona sandboxes and Docker containers."""
    print("\n" + "="*80)
    print("CURRENT RESOURCES")
    print("="*80 + "\n")

    # List Daytona sandboxes
    api_key = os.environ.get("DAYTONA_API_KEY")
    if api_key:
        try:
            manager = DaytonaCleanupManager(api_key=api_key)
            sandboxes = manager.list_sandboxes()
            print(f"Daytona Sandboxes: {len(sandboxes)}")
            for sandbox in sandboxes[:10]:  # Show first 10
                sandbox_id = getattr(sandbox, 'id', 'N/A')
                status = getattr(sandbox, 'status', 'N/A')
                print(f"  - {sandbox_id} ({status})")
            if len(sandboxes) > 10:
                print(f"  ... and {len(sandboxes) - 10} more")
            print()
        except Exception as e:
            print(f"Could not list Daytona sandboxes: {e}\n")
    else:
        print("DAYTONA_API_KEY not set - skipping Daytona listing\n")

    # List Docker containers
    try:
        manager = DockerCleanupManager()
        containers = manager.list_containers(all_containers=True)
        print(f"Docker Containers: {len(containers)}")
        for cid in containers[:10]:  # Show first 10
            print(f"  - {cid}")
        if len(containers) > 10:
            print(f"  ... and {len(containers) - 10} more")
        print()
    except Exception as e:
        print(f"Could not list Docker containers: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive cleanup for Daytona and Docker containers"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all resources without cleanup"
    )
    parser.add_argument(
        "--daytona-only",
        action="store_true",
        help="Only cleanup Daytona sandboxes"
    )
    parser.add_argument(
        "--docker-only",
        action="store_true",
        help="Only cleanup Docker containers"
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Aggressive cleanup (stopped/exited + old resources)"
    )
    parser.add_argument(
        "--daytona-days",
        type=int,
        default=1,
        help="Age threshold in days for Daytona cleanup (default: 1)"
    )
    parser.add_argument(
        "--docker-hours",
        type=int,
        default=24,
        help="Age threshold in hours for Docker cleanup (default: 24)"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first cleanup failure"
    )

    args = parser.parse_args()

    if not CLEANUP_MANAGER_AVAILABLE:
        print("Error: cleanup_manager module not available")
        sys.exit(1)

    # List mode
    if args.list:
        list_resources()
        return

    print("\n" + "="*80)
    print("COMPREHENSIVE CLEANUP - DAYTONA & DOCKER")
    print("="*80)

    total_daytona_deleted = 0
    total_daytona_failed = 0
    total_docker_deleted = 0
    total_docker_failed = 0

    # Daytona cleanup
    if not args.docker_only:
        daytona_deleted, daytona_failed = cleanup_daytona(
            max_age_days=args.daytona_days,
            cleanup_stopped=args.aggressive or True,
            fail_fast=args.fail_fast
        )
        total_daytona_deleted += daytona_deleted
        total_daytona_failed += daytona_failed

    # Docker cleanup
    if not args.daytona_only:
        docker_deleted, docker_failed = cleanup_docker(
            max_age_hours=args.docker_hours,
            cleanup_exited=args.aggressive or True,
            fail_fast=args.fail_fast
        )
        total_docker_deleted += docker_deleted
        total_docker_failed += docker_failed

    # Summary
    print("\n" + "="*80)
    print("CLEANUP SUMMARY")
    print("="*80)
    print(f"\nDaytona: {total_daytona_deleted} deleted, {total_daytona_failed} failed")
    print(f"Docker:  {total_docker_deleted} deleted, {total_docker_failed} failed")
    print(f"\nTotal:   {total_daytona_deleted + total_docker_deleted} deleted, "
          f"{total_daytona_failed + total_docker_failed} failed")

    if total_daytona_failed + total_docker_failed > 0:
        print("\n⚠️  WARNING: Some cleanup operations failed!")
        print("Run with --fail-fast to stop on first failure for debugging.")
        sys.exit(1)
    else:
        print("\n✓ All cleanup operations completed successfully!")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
