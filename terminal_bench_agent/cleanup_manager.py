"""Centralized cleanup manager for Daytona and Docker containers.

This module provides robust cleanup management with:
- Guaranteed cleanup via context managers
- Retry logic with exponential backoff
- Proper error handling and logging
- Support for both Daytona and Docker containers
"""

import asyncio
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Callable

try:
    from daytona import Daytona
    DAYTONA_AVAILABLE = True
except ImportError:
    DAYTONA_AVAILABLE = False


logger = logging.getLogger(__name__)


class CleanupError(Exception):
    """Exception raised when cleanup fails critically."""
    pass


class RetryableCleanupError(Exception):
    """Exception raised when cleanup fails but can be retried."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries} retry attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper
    return decorator


async def async_retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Async decorator for retry logic with exponential backoff."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries} retry attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper
    return decorator


class DaytonaCleanupManager:
    """Manager for Daytona sandbox cleanup with retry logic."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Daytona cleanup manager.

        Args:
            api_key: Daytona API key (uses DAYTONA_API_KEY env var if not provided)
        """
        if not DAYTONA_AVAILABLE:
            raise ImportError("Daytona SDK not available. Install with: pip install daytona-sdk")

        self.api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if not self.api_key:
            raise ValueError("DAYTONA_API_KEY not found in environment")

        self.client = None
        self._connect()

    def _connect(self):
        """Connect to Daytona API."""
        try:
            self.client = Daytona()
            logger.info("Connected to Daytona API")
        except Exception as e:
            logger.error(f"Failed to connect to Daytona: {e}")
            raise CleanupError(f"Daytona connection failed: {e}")

    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    def delete_sandbox(self, sandbox_id: str) -> bool:
        """Delete a single sandbox with retry logic.

        Args:
            sandbox_id: ID of sandbox to delete

        Returns:
            True if deletion successful

        Raises:
            RetryableCleanupError: If deletion fails after all retries
        """
        try:
            self.client.delete(sandbox_id)
            logger.info(f"Deleted Daytona sandbox: {sandbox_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete sandbox {sandbox_id}: {e}")
            raise RetryableCleanupError(f"Sandbox deletion failed: {e}")

    def list_sandboxes(self) -> List:
        """List all sandboxes.

        Returns:
            List of sandbox objects
        """
        try:
            result = self.client.list()
            sandboxes = getattr(result, 'items', [])
            logger.info(f"Found {len(sandboxes)} Daytona sandboxes")
            return sandboxes
        except Exception as e:
            logger.error(f"Failed to list sandboxes: {e}")
            return []

    def cleanup_old_sandboxes(
        self,
        max_age_days: int = 1,
        fail_fast: bool = False
    ) -> tuple[int, int]:
        """Clean up sandboxes older than specified days.

        Args:
            max_age_days: Delete sandboxes older than this many days
            fail_fast: If True, stop on first failure; if False, continue

        Returns:
            Tuple of (deleted_count, failed_count)
        """
        cutoff_date = datetime.now(datetime.now().astimezone().tzinfo) - timedelta(days=max_age_days)
        deleted_count = 0
        failed_count = 0

        sandboxes = self.list_sandboxes()
        logger.info(f"Cleaning up sandboxes older than {max_age_days} days (cutoff: {cutoff_date})")

        for sandbox in sandboxes:
            try:
                created = getattr(sandbox, 'created_at', None)
                if not created:
                    continue

                # Parse timestamp
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                elif isinstance(created, datetime):
                    created_dt = created
                else:
                    continue

                # Delete if old enough
                if created_dt < cutoff_date:
                    sandbox_id = getattr(sandbox, 'id', 'unknown')
                    try:
                        self.delete_sandbox(sandbox_id)
                        deleted_count += 1
                    except RetryableCleanupError as e:
                        failed_count += 1
                        if fail_fast:
                            raise CleanupError(f"Cleanup failed on sandbox {sandbox_id}: {e}")

            except Exception as e:
                logger.error(f"Error processing sandbox: {e}")
                failed_count += 1
                if fail_fast:
                    raise

        logger.info(f"Deleted {deleted_count} old sandboxes, {failed_count} failures")
        return deleted_count, failed_count

    def cleanup_stopped_sandboxes(self, fail_fast: bool = False) -> tuple[int, int]:
        """Clean up stopped/failed/terminated sandboxes.

        Args:
            fail_fast: If True, stop on first failure; if False, continue

        Returns:
            Tuple of (deleted_count, failed_count)
        """
        deleted_count = 0
        failed_count = 0

        sandboxes = self.list_sandboxes()
        stopped_states = ['stopped', 'terminated', 'failed', 'error', 'exited']
        logger.info(f"Cleaning up sandboxes in states: {stopped_states}")

        for sandbox in sandboxes:
            try:
                status = getattr(sandbox, 'status', '').lower()
                if status in stopped_states:
                    sandbox_id = getattr(sandbox, 'id', 'unknown')
                    try:
                        self.delete_sandbox(sandbox_id)
                        deleted_count += 1
                    except RetryableCleanupError as e:
                        failed_count += 1
                        if fail_fast:
                            raise CleanupError(f"Cleanup failed on sandbox {sandbox_id}: {e}")

            except Exception as e:
                logger.error(f"Error processing sandbox: {e}")
                failed_count += 1
                if fail_fast:
                    raise

        logger.info(f"Deleted {deleted_count} stopped sandboxes, {failed_count} failures")
        return deleted_count, failed_count


class DockerCleanupManager:
    """Manager for Docker container cleanup."""

    @staticmethod
    def _run_docker_command(args: List[str]) -> subprocess.CompletedProcess:
        """Run a docker command.

        Args:
            args: Docker command arguments

        Returns:
            Completed process result
        """
        cmd = ["docker"] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result
        except subprocess.TimeoutExpired:
            logger.error(f"Docker command timed out: {' '.join(cmd)}")
            raise CleanupError("Docker command timeout")
        except FileNotFoundError:
            logger.error("Docker command not found - is Docker installed?")
            raise CleanupError("Docker not available")

    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    def stop_container(self, container_id: str, timeout: int = 10) -> bool:
        """Stop a Docker container with retry logic.

        Args:
            container_id: Container ID or name
            timeout: Timeout in seconds for graceful stop

        Returns:
            True if successful
        """
        try:
            result = self._run_docker_command(["stop", "-t", str(timeout), container_id])
            if result.returncode == 0:
                logger.info(f"Stopped Docker container: {container_id}")
                return True
            else:
                logger.error(f"Failed to stop container {container_id}: {result.stderr}")
                raise RetryableCleanupError(f"Stop failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error stopping container {container_id}: {e}")
            raise

    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    def remove_container(self, container_id: str, force: bool = True) -> bool:
        """Remove a Docker container with retry logic.

        Args:
            container_id: Container ID or name
            force: Force removal even if running

        Returns:
            True if successful
        """
        try:
            args = ["rm"]
            if force:
                args.append("-f")
            args.append(container_id)

            result = self._run_docker_command(args)
            if result.returncode == 0:
                logger.info(f"Removed Docker container: {container_id}")
                return True
            else:
                logger.error(f"Failed to remove container {container_id}: {result.stderr}")
                raise RetryableCleanupError(f"Remove failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error removing container {container_id}: {e}")
            raise

    def list_containers(
        self,
        all_containers: bool = True,
        filters: Optional[dict] = None
    ) -> List[str]:
        """List Docker containers.

        Args:
            all_containers: Include stopped containers
            filters: Docker filters (e.g., {"label": "terminal-bench"})

        Returns:
            List of container IDs
        """
        args = ["ps", "-q"]
        if all_containers:
            args.append("-a")

        if filters:
            for key, value in filters.items():
                args.extend(["--filter", f"{key}={value}"])

        try:
            result = self._run_docker_command(args)
            if result.returncode == 0:
                container_ids = [cid.strip() for cid in result.stdout.split('\n') if cid.strip()]
                logger.info(f"Found {len(container_ids)} Docker containers")
                return container_ids
            else:
                logger.error(f"Failed to list containers: {result.stderr}")
                return []
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
            return []

    def cleanup_containers(
        self,
        filters: Optional[dict] = None,
        stop_timeout: int = 10,
        fail_fast: bool = False
    ) -> tuple[int, int]:
        """Clean up Docker containers matching filters.

        Args:
            filters: Docker filters (e.g., {"status": "exited"})
            stop_timeout: Timeout for graceful stop
            fail_fast: If True, stop on first failure

        Returns:
            Tuple of (deleted_count, failed_count)
        """
        deleted_count = 0
        failed_count = 0

        containers = self.list_containers(all_containers=True, filters=filters)
        logger.info(f"Cleaning up {len(containers)} Docker containers")

        for container_id in containers:
            try:
                # Try to stop first (in case it's running)
                try:
                    self.stop_container(container_id, timeout=stop_timeout)
                except:
                    pass  # Container might already be stopped

                # Remove container
                self.remove_container(container_id, force=True)
                deleted_count += 1

            except RetryableCleanupError as e:
                failed_count += 1
                if fail_fast:
                    raise CleanupError(f"Cleanup failed on container {container_id}: {e}")
            except Exception as e:
                logger.error(f"Error cleaning container {container_id}: {e}")
                failed_count += 1
                if fail_fast:
                    raise

        logger.info(f"Cleaned {deleted_count} Docker containers, {failed_count} failures")
        return deleted_count, failed_count

    def cleanup_old_containers(
        self,
        max_age_hours: int = 24,
        fail_fast: bool = False
    ) -> tuple[int, int]:
        """Clean up containers older than specified hours.

        Args:
            max_age_hours: Delete containers older than this many hours
            fail_fast: If True, stop on first failure

        Returns:
            Tuple of (deleted_count, failed_count)
        """
        # Get containers created before cutoff time
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cutoff_str = cutoff_time.strftime("%Y-%m-%dT%H:%M:%S")

        # Docker filter for containers created before cutoff
        filters = {"before": cutoff_str}

        return self.cleanup_containers(filters=filters, fail_fast=fail_fast)


@asynccontextmanager
async def managed_daytona_sandbox(
    sandbox_id: str,
    cleanup_manager: Optional[DaytonaCleanupManager] = None,
    cleanup_on_error: bool = True
):
    """Context manager for guaranteed Daytona sandbox cleanup.

    Args:
        sandbox_id: ID of the sandbox to manage
        cleanup_manager: Optional cleanup manager instance
        cleanup_on_error: Clean up even if exception occurs

    Yields:
        Sandbox ID

    Example:
        async with managed_daytona_sandbox("sandbox-123") as sandbox_id:
            # Use sandbox
            pass
        # Sandbox is automatically cleaned up
    """
    manager = cleanup_manager or DaytonaCleanupManager()

    try:
        logger.info(f"Managing Daytona sandbox: {sandbox_id}")
        yield sandbox_id
    except Exception as e:
        logger.error(f"Error in sandbox {sandbox_id}: {e}")
        if cleanup_on_error:
            try:
                manager.delete_sandbox(sandbox_id)
                logger.info(f"Cleaned up sandbox {sandbox_id} after error")
            except Exception as cleanup_err:
                logger.error(f"Failed to cleanup sandbox {sandbox_id}: {cleanup_err}")
        raise
    finally:
        # Always attempt cleanup
        try:
            manager.delete_sandbox(sandbox_id)
            logger.info(f"Cleaned up sandbox {sandbox_id} (normal exit)")
        except Exception as e:
            logger.warning(f"Cleanup warning for sandbox {sandbox_id}: {e}")


@contextmanager
def managed_docker_container(
    container_id: str,
    cleanup_manager: Optional[DockerCleanupManager] = None,
    cleanup_on_error: bool = True,
    stop_timeout: int = 10
):
    """Context manager for guaranteed Docker container cleanup.

    Args:
        container_id: ID of container to manage
        cleanup_manager: Optional cleanup manager instance
        cleanup_on_error: Clean up even if exception occurs
        stop_timeout: Timeout for graceful stop

    Yields:
        Container ID

    Example:
        with managed_docker_container("container-123") as cid:
            # Use container
            pass
        # Container is automatically cleaned up
    """
    manager = cleanup_manager or DockerCleanupManager()

    try:
        logger.info(f"Managing Docker container: {container_id}")
        yield container_id
    except Exception as e:
        logger.error(f"Error in container {container_id}: {e}")
        if cleanup_on_error:
            try:
                manager.stop_container(container_id, timeout=stop_timeout)
                manager.remove_container(container_id, force=True)
                logger.info(f"Cleaned up container {container_id} after error")
            except Exception as cleanup_err:
                logger.error(f"Failed to cleanup container {container_id}: {cleanup_err}")
        raise
    finally:
        # Always attempt cleanup
        try:
            manager.stop_container(container_id, timeout=stop_timeout)
            manager.remove_container(container_id, force=True)
            logger.info(f"Cleaned up container {container_id} (normal exit)")
        except Exception as e:
            logger.warning(f"Cleanup warning for container {container_id}: {e}")
