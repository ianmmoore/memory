"""
Usage tracking middleware for billing and analytics.

Tracks:
- API calls per endpoint
- Tokens consumed (input/output)
- Memories read/written
- Embeddings generated
- Processing time
- Estimated cost

Data is buffered in Redis and periodically flushed to PostgreSQL.
"""

import time
import asyncio
from datetime import datetime, timezone
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
import json

from fastapi import Request, Response
from redis import asyncio as aioredis
import structlog

from config.settings import settings

logger = structlog.get_logger()


@dataclass
class UsageRecord:
    """Individual usage record for a request."""
    org_id: str
    api_key_id: Optional[str]
    timestamp: str
    endpoint: str
    method: str
    status_code: int = 200
    processing_time_ms: int = 0

    # Usage metrics
    tokens_input: int = 0
    tokens_output: int = 0
    memories_read: int = 0
    memories_written: int = 0
    embeddings_generated: int = 0

    # Cost tracking (in cents)
    cost_cents: int = 0

    # Request metadata
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class UsageTracker:
    """
    Tracks API usage and buffers in Redis for efficient bulk writes.

    Usage data is:
    1. Written to Redis list on each request
    2. Periodically flushed to PostgreSQL by background worker
    3. Rolled up daily for efficient billing queries
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self._redis: Optional[aioredis.Redis] = None
        self.buffer_key = f"{settings.redis_prefix}usage:buffer"
        self.realtime_key_prefix = f"{settings.redis_prefix}usage:realtime"

    async def get_redis(self) -> aioredis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def track(self, record: UsageRecord):
        """
        Track a usage event.

        Writes to:
        1. Buffer list (for batch DB writes)
        2. Real-time counters (for quota checks)
        """
        try:
            redis = await self.get_redis()

            # Serialize record
            record_json = json.dumps(record.to_dict())

            # Add to buffer for batch processing
            await redis.lpush(self.buffer_key, record_json)

            # Update real-time counters for current period
            period_key = datetime.now(timezone.utc).strftime("%Y-%m")
            realtime_key = f"{self.realtime_key_prefix}:{record.org_id}:{period_key}"

            # Increment counters atomically
            pipe = redis.pipeline()
            pipe.hincrby(realtime_key, "api_calls", 1)
            pipe.hincrby(realtime_key, "tokens_input", record.tokens_input)
            pipe.hincrby(realtime_key, "tokens_output", record.tokens_output)
            pipe.hincrby(realtime_key, "memories_read", record.memories_read)
            pipe.hincrby(realtime_key, "memories_written", record.memories_written)
            pipe.hincrby(realtime_key, "embeddings", record.embeddings_generated)
            pipe.hincrby(realtime_key, "cost_cents", record.cost_cents)
            pipe.expire(realtime_key, 86400 * 35)  # Keep for ~35 days
            await pipe.execute()

        except Exception as e:
            logger.error("usage_tracking_failed", error=str(e), org_id=record.org_id)

    async def get_current_usage(self, org_id: str) -> dict:
        """
        Get real-time usage for current billing period.
        """
        try:
            redis = await self.get_redis()
            period_key = datetime.now(timezone.utc).strftime("%Y-%m")
            realtime_key = f"{self.realtime_key_prefix}:{org_id}:{period_key}"

            data = await redis.hgetall(realtime_key)

            return {
                "api_calls": int(data.get("api_calls", 0)),
                "tokens_input": int(data.get("tokens_input", 0)),
                "tokens_output": int(data.get("tokens_output", 0)),
                "tokens_total": int(data.get("tokens_input", 0)) + int(data.get("tokens_output", 0)),
                "memories_read": int(data.get("memories_read", 0)),
                "memories_written": int(data.get("memories_written", 0)),
                "embeddings": int(data.get("embeddings", 0)),
                "cost_cents": int(data.get("cost_cents", 0)),
                "period": period_key,
            }

        except Exception as e:
            logger.error("get_current_usage_failed", error=str(e))
            return {}

    async def flush_to_database(self, batch_size: int = 100) -> int:
        """
        Flush buffered usage records to PostgreSQL.

        Called by background worker. Returns number of records flushed.
        """
        from db.database import get_db_context
        from db.models import UsageEvent

        try:
            redis = await self.get_redis()

            # Pop records from buffer
            records = []
            for _ in range(batch_size):
                record_json = await redis.rpop(self.buffer_key)
                if not record_json:
                    break
                records.append(json.loads(record_json))

            if not records:
                return 0

            # Bulk insert to database
            async with get_db_context() as db:
                for record in records:
                    event = UsageEvent(
                        org_id=record["org_id"],
                        api_key_id=record.get("api_key_id"),
                        timestamp=datetime.fromisoformat(record["timestamp"]),
                        endpoint=record["endpoint"],
                        method=record["method"],
                        tokens_input=record.get("tokens_input", 0),
                        tokens_output=record.get("tokens_output", 0),
                        memories_read=record.get("memories_read", 0),
                        memories_written=record.get("memories_written", 0),
                        embeddings_generated=record.get("embeddings_generated", 0),
                        processing_time_ms=record.get("processing_time_ms", 0),
                        status_code=record.get("status_code", 200),
                        cost_cents=record.get("cost_cents", 0),
                        request_id=record.get("request_id"),
                        ip_address=record.get("ip_address"),
                        user_agent=record.get("user_agent"),
                    )
                    db.add(event)

                await db.commit()

            logger.info("usage_flushed_to_db", count=len(records))
            return len(records)

        except Exception as e:
            logger.error("usage_flush_failed", error=str(e))
            return 0


# Global usage tracker
usage_tracker = UsageTracker()


def calculate_cost(
    tokens_input: int = 0,
    tokens_output: int = 0,
    embeddings: int = 0,
) -> int:
    """
    Calculate internal cost in cents for a request.

    Based on OpenAI pricing (adjust as needed):
    - gpt-4o-mini input: $0.15 per 1M tokens
    - gpt-4o-mini output: $0.60 per 1M tokens
    - text-embedding-3-small: $0.02 per 1M tokens
    """
    # Convert to cents per million tokens
    input_cost = (tokens_input / 1_000_000) * 15  # 15 cents per 1M
    output_cost = (tokens_output / 1_000_000) * 60  # 60 cents per 1M
    embedding_cost = (embeddings * 1536 / 1_000_000) * 2  # ~1536 tokens per embedding

    # Add infrastructure cost (amortized)
    infra_cost = 0.001  # $0.00001 per request

    total_cents = input_cost + output_cost + embedding_cost + infra_cost
    return int(total_cents * 100)  # Return in hundredths of cents for precision


async def track_usage(
    request: Request,
    response: Response,
    tokens_input: int = 0,
    tokens_output: int = 0,
    memories_read: int = 0,
    memories_written: int = 0,
    embeddings_generated: int = 0,
):
    """
    Helper function to track usage after request processing.

    Call this at the end of endpoint handlers that consume billable resources.
    """
    org_id = getattr(request.state, "org_id", None)
    api_key_id = getattr(request.state, "api_key_id", None)

    if not org_id:
        return

    # Calculate processing time
    start_time = getattr(request.state, "start_time", time.time())
    processing_time_ms = int((time.time() - start_time) * 1000)

    # Calculate cost
    cost_cents = calculate_cost(tokens_input, tokens_output, embeddings_generated)

    record = UsageRecord(
        org_id=org_id,
        api_key_id=api_key_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        processing_time_ms=processing_time_ms,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        memories_read=memories_read,
        memories_written=memories_written,
        embeddings_generated=embeddings_generated,
        cost_cents=cost_cents,
        request_id=getattr(request.state, "request_id", None),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent", "")[:500],
    )

    await usage_tracker.track(record)


class UsageTrackingMiddleware:
    """
    ASGI middleware that tracks request timing and basic usage.

    For detailed usage (tokens, memories), endpoints should call track_usage().
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Record start time
        scope.setdefault("state", {})
        scope["state"]["start_time"] = time.time()

        # Generate request ID
        import uuid
        scope["state"]["request_id"] = str(uuid.uuid4())

        # Track response status
        status_code = 200

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)

                # Add request ID header
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", scope["state"]["request_id"].encode()))
                message["headers"] = headers

            await send(message)

        await self.app(scope, receive, send_wrapper)

        # Track basic request (no token info - that's added by endpoints)
        org_id = scope["state"].get("org_id")
        if org_id:
            processing_time = int((time.time() - scope["state"]["start_time"]) * 1000)

            record = UsageRecord(
                org_id=org_id,
                api_key_id=scope["state"].get("api_key_id"),
                timestamp=datetime.now(timezone.utc).isoformat(),
                endpoint=scope["path"],
                method=scope["method"],
                status_code=status_code,
                processing_time_ms=processing_time,
                request_id=scope["state"]["request_id"],
            )

            # Non-blocking track
            asyncio.create_task(usage_tracker.track(record))


class UsageFlushWorker:
    """
    Background worker that periodically flushes usage to database.
    Run this as a separate process or in a background task.
    """

    def __init__(
        self,
        flush_interval: int = None,
        batch_size: int = None,
    ):
        self.flush_interval = flush_interval or settings.usage_flush_interval_seconds
        self.batch_size = batch_size or settings.usage_batch_size
        self.running = False

    async def start(self):
        """Start the flush worker."""
        self.running = True
        logger.info("usage_flush_worker_started")

        while self.running:
            try:
                count = await usage_tracker.flush_to_database(self.batch_size)
                if count > 0:
                    logger.debug("usage_flushed", count=count)
            except Exception as e:
                logger.error("usage_flush_error", error=str(e))

            await asyncio.sleep(self.flush_interval)

    def stop(self):
        """Stop the flush worker."""
        self.running = False
        logger.info("usage_flush_worker_stopped")
