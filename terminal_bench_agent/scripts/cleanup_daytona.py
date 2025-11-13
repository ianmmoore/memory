#!/usr/bin/env python3
"""Script to cleanup old Daytona sandboxes to avoid disk limit issues.

This script lists all sandboxes and optionally archives/deletes old ones
to free up storage space.

Usage:
    # List all sandboxes
    python scripts/cleanup_daytona.py --list

    # Delete sandboxes older than 1 day
    python scripts/cleanup_daytona.py --delete-old --days 1 --no-dry-run

    # Delete all stopped sandboxes
    python scripts/cleanup_daytona.py --delete-stopped --no-dry-run

    # Delete all sandboxes (careful!)
    python scripts/cleanup_daytona.py --delete-all --confirm --no-dry-run
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    from daytona import Daytona
    DAYTONA_AVAILABLE = True
except ImportError:
    DAYTONA_AVAILABLE = False
    print("Error: Daytona SDK not installed. Install with: pip install daytona-sdk")
    sys.exit(1)


def list_sandboxes(client: Daytona) -> list:
    """List all sandboxes.

    Args:
        client: Daytona client instance

    Returns:
        List of sandbox objects
    """
    try:
        result = client.list()
        # The result has an 'items' property with the list of sandboxes
        return getattr(result, 'items', [])
    except Exception as e:
        print(f"Error listing sandboxes: {e}")
        return []


def print_sandbox_info(sandboxes: list):
    """Print information about sandboxes.

    Args:
        sandboxes: List of sandbox objects
    """
    if not sandboxes:
        print("No sandboxes found.")
        return

    print(f"\nFound {len(sandboxes)} sandbox(es):\n")
    print(f"{'ID':<40} {'Status':<12} {'Created':<20}")
    print("-" * 80)

    for sandbox in sandboxes:
        sandbox_id = getattr(sandbox, 'id', 'N/A')
        status = getattr(sandbox, 'status', 'N/A')
        created = getattr(sandbox, 'created_at', 'N/A')

        # Format created timestamp if available
        if created != 'N/A' and isinstance(created, (str, datetime)):
            if isinstance(created, str):
                try:
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    created = created_dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass

        print(f"{sandbox_id:<40} {status:<12} {created:<20}")

    print()


def delete_old_sandboxes(client: Daytona, days: int, dry_run: bool = True) -> int:
    """Delete sandboxes older than specified days.

    Args:
        client: Daytona client instance
        days: Age threshold in days
        dry_run: If True, only print what would be deleted

    Returns:
        Number of sandboxes deleted
    """
    sandboxes = list_sandboxes(client)
    cutoff_date = datetime.now(datetime.now().astimezone().tzinfo) - timedelta(days=days)
    deleted_count = 0

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

            # Check if older than cutoff
            if created_dt < cutoff_date:
                sandbox_id = getattr(sandbox, 'id', 'unknown')

                if dry_run:
                    print(f"Would delete: {sandbox_id} (created {created_dt})")
                else:
                    print(f"Deleting: {sandbox_id}...")
                    try:
                        client.delete(sandbox_id)
                        print(f"  ✓ Deleted {sandbox_id}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to delete {sandbox_id}: {e}")

        except Exception as e:
            print(f"Error processing sandbox: {e}")
            continue

    return deleted_count


def delete_stopped_sandboxes(client: Daytona, dry_run: bool = True) -> int:
    """Delete all stopped sandboxes.

    Args:
        client: Daytona client instance
        dry_run: If True, only print what would be deleted

    Returns:
        Number of sandboxes deleted
    """
    sandboxes = list_sandboxes(client)
    deleted_count = 0

    for sandbox in sandboxes:
        try:
            status = getattr(sandbox, 'status', '').lower()
            sandbox_id = getattr(sandbox, 'id', 'unknown')

            if status in ['stopped', 'terminated', 'failed', 'error', 'exited']:
                if dry_run:
                    print(f"Would delete: {sandbox_id} (status: {status})")
                else:
                    print(f"Deleting: {sandbox_id}...")
                    try:
                        client.delete(sandbox_id)
                        print(f"  ✓ Deleted {sandbox_id}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to delete {sandbox_id}: {e}")

        except Exception as e:
            print(f"Error processing sandbox: {e}")
            continue

    return deleted_count


def delete_all_sandboxes(client: Daytona, confirmed: bool = False) -> int:
    """Delete ALL sandboxes (use with caution!).

    Args:
        client: Daytona client instance
        confirmed: Must be True to actually delete

    Returns:
        Number of sandboxes deleted
    """
    if not confirmed:
        print("Error: Must confirm deletion with --confirm flag")
        return 0

    sandboxes = list_sandboxes(client)
    deleted_count = 0

    print(f"\n⚠️  WARNING: Deleting ALL {len(sandboxes)} sandboxes!")

    for sandbox in sandboxes:
        try:
            sandbox_id = getattr(sandbox, 'id', 'unknown')
            print(f"Deleting: {sandbox_id}...")

            try:
                client.delete(sandbox_id)
                print(f"  ✓ Deleted {sandbox_id}")
                deleted_count += 1
            except Exception as e:
                print(f"  ✗ Failed to delete {sandbox_id}: {e}")

        except Exception as e:
            print(f"Error processing sandbox: {e}")
            continue

    return deleted_count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cleanup Daytona sandboxes to free up storage"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all sandboxes"
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Delete old sandboxes"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Age threshold in days for deletion (default: 1)"
    )
    parser.add_argument(
        "--delete-stopped",
        action="store_true",
        help="Delete all stopped/failed sandboxes"
    )
    parser.add_argument(
        "--delete-all",
        action="store_true",
        help="Delete ALL sandboxes (requires --confirm)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm destructive operations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be done without actually doing it (default: True)"
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually perform the operations (disables dry-run)"
    )

    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        print("Error: DAYTONA_API_KEY environment variable not set")
        sys.exit(1)

    # Determine dry-run mode
    dry_run = not args.no_dry_run

    # Initialize Daytona client
    print("Connecting to Daytona...")
    try:
        # Daytona reads from DAYTONA_API_KEY environment variable automatically
        client = Daytona()
    except Exception as e:
        print(f"Error connecting to Daytona: {e}")
        sys.exit(1)

    # Execute requested action
    if args.list or not any([args.delete_old, args.delete_stopped, args.delete_all]):
        # Default action: list sandboxes
        sandboxes = list_sandboxes(client)
        print_sandbox_info(sandboxes)

        # Print storage info if available
        print("To free up storage:")
        print("  - Delete old sandboxes: python cleanup_daytona.py --delete-old --days 1 --no-dry-run")
        print("  - Delete stopped: python cleanup_daytona.py --delete-stopped --no-dry-run")

    elif args.delete_old:
        print(f"\nDeleting sandboxes older than {args.days} day(s)...")
        if dry_run:
            print("(DRY RUN - use --no-dry-run to actually delete)\n")
        count = delete_old_sandboxes(client, args.days, dry_run)
        print(f"\n{'Would delete' if dry_run else 'Deleted'} {count} sandbox(es)")

    elif args.delete_stopped:
        print("\nDeleting stopped/failed sandboxes...")
        if dry_run:
            print("(DRY RUN - use --no-dry-run to actually delete)\n")
        count = delete_stopped_sandboxes(client, dry_run)
        print(f"\n{'Would delete' if dry_run else 'Deleted'} {count} sandbox(es)")

    elif args.delete_all:
        if not args.confirm:
            print("\nError: Deleting all sandboxes requires --confirm flag")
            print("Usage: python cleanup_daytona.py --delete-all --confirm --no-dry-run")
            sys.exit(1)

        count = delete_all_sandboxes(client, args.confirm)
        print(f"\nDeleted {count} sandbox(es)")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
