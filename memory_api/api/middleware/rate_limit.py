"""
Rate limiting middleware using Redis sliding window algorithm.

Features:
- Per-organization rate limits based on plan tier
- Sliding window algorithm for smooth rate limiting
- Concurrent request limits
- Daily request quotas
- Returns rate limit headers on all responses
"""

import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from redis import asyncio as aioredis
import structlog

from config.settings import settings
from db.models import Organization
from .auth import get_current_org, CurrentOrg

logger = structlog.get_logger()


class RateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm.

    The sliding window provides smoother rate limiting than fixed windows
    by considering the overlap between the previous and current window.
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self._redis: Optional[aioredis.Redis] = None

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

    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int = 60,
    ) -> tuple[bool, dict]:
        """
        Check if request is within rate limit.

        Args:
            identifier: Unique identifier (e.g., org_id or API key)
            limit: Maximum requests allowed in window
            window_seconds: Time window in seconds

        Returns:
            tuple: (allowed: bool, info: dict with limit details)
        """
        redis = await self.get_redis()
        now = time.time()
        key = f"{settings.redis_prefix}ratelimit:{identifier}"

        # Use a Lua script for atomic operation
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])

        -- Remove old entries outside the window
        redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)

        -- Count current requests in window
        local current = redis.call('ZCARD', key)

        if current < limit then
            -- Add new request
            redis.call('ZADD', key, now, now .. '-' .. math.random())
            redis.call('EXPIRE', key, window)
            return {1, current + 1, limit}
        else
            return {0, current, limit}
        end
        """

        try:
            result = await redis.eval(
                lua_script,
                1,
                key,
                str(now),
                str(window_seconds),
                str(limit),
            )

            allowed = bool(result[0])
            current = int(result[1])
            max_limit = int(result[2])

            # Calculate reset time
            reset_time = int(now) + window_seconds

            return allowed, {
                "limit": max_limit,
                "remaining": max(0, max_limit - current),
                "reset": reset_time,
                "retry_after": window_seconds if not allowed else None,
            }

        except Exception as e:
            logger.error("rate_limit_check_failed", error=str(e))
            # Fail open - allow request if Redis is down
            return True, {
                "limit": limit,
                "remaining": limit,
                "reset": int(now) + window_seconds,
                "retry_after": None,
            }

    async def check_daily_quota(
        self,
        identifier: str,
        daily_limit: Optional[int],
    ) -> tuple[bool, dict]:
        """
        Check daily request quota.

        Args:
            identifier: Unique identifier
            daily_limit: Maximum requests per day (None = unlimited)

        Returns:
            tuple: (allowed: bool, info: dict)
        """
        if daily_limit is None:
            return True, {"daily_limit": None, "daily_used": 0}

        redis = await self.get_redis()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"{settings.redis_prefix}daily:{identifier}:{today}"

        try:
            current = await redis.incr(key)

            # Set expiry on first request of the day
            if current == 1:
                await redis.expire(key, 86400 + 3600)  # 25 hours

            allowed = current <= daily_limit

            return allowed, {
                "daily_limit": daily_limit,
                "daily_used": current,
                "daily_remaining": max(0, daily_limit - current),
            }

        except Exception as e:
            logger.error("daily_quota_check_failed", error=str(e))
            return True, {"daily_limit": daily_limit, "daily_used": 0}

    async def check_concurrent(
        self,
        identifier: str,
        max_concurrent: int,
        request_id: str,
    ) -> tuple[bool, dict]:
        """
        Check concurrent request limit.

        Args:
            identifier: Unique identifier
            max_concurrent: Maximum concurrent requests
            request_id: Unique request identifier

        Returns:
            tuple: (allowed: bool, info: dict)
        """
        redis = await self.get_redis()
        key = f"{settings.redis_prefix}concurrent:{identifier}"

        try:
            # Add request with TTL (auto-cleanup for stuck requests)
            current = await redis.sadd(key, request_id)
            count = await redis.scard(key)
            await redis.expire(key, 300)  # 5 minute max request time

            allowed = count <= max_concurrent

            if not allowed:
                await redis.srem(key, request_id)

            return allowed, {
                "concurrent_limit": max_concurrent,
                "concurrent_current": count,
            }

        except Exception as e:
            logger.error("concurrent_check_failed", error=str(e))
            return True, {"concurrent_limit": max_concurrent}

    async def release_concurrent(self, identifier: str, request_id: str):
        """Release concurrent request slot."""
        try:
            redis = await self.get_redis()
            key = f"{settings.redis_prefix}concurrent:{identifier}"
            await redis.srem(key, request_id)
        except Exception as e:
            logger.error("concurrent_release_failed", error=str(e))


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_rate_limits_for_plan(plan_id: str) -> dict:
    """Get rate limits based on plan tier."""
    limits = {
        "free": {
            "requests_per_minute": settings.rate_limit_free,
            "requests_per_day": 100,
            "concurrent": 2,
        },
        "starter": {
            "requests_per_minute": settings.rate_limit_starter,
            "requests_per_day": 5000,
            "concurrent": 5,
        },
        "professional": {
            "requests_per_minute": settings.rate_limit_professional,
            "requests_per_day": 50000,
            "concurrent": 20,
        },
        "enterprise": {
            "requests_per_minute": settings.rate_limit_enterprise,
            "requests_per_day": None,  # Unlimited
            "concurrent": 100,
        },
    }
    return limits.get(plan_id, limits["free"])


async def check_rate_limit(
    request: Request,
    org: Organization = Depends(get_current_org),
) -> dict:
    """
    FastAPI dependency for rate limit checking.

    Raises HTTPException if rate limit exceeded.
    Returns rate limit info for response headers.
    """
    limits = get_rate_limits_for_plan(org.plan_id)
    request_id = getattr(request.state, "request_id", str(time.time()))

    # Check per-minute rate limit
    allowed, minute_info = await rate_limiter.check_rate_limit(
        identifier=org.id,
        limit=limits["requests_per_minute"],
        window_seconds=60,
    )

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "RATE_LIMIT_EXCEEDED",
                "message": f"Rate limit of {limits['requests_per_minute']} requests/minute exceeded",
                "retry_after": minute_info["retry_after"],
            },
            headers={
                "X-RateLimit-Limit": str(minute_info["limit"]),
                "X-RateLimit-Remaining": str(minute_info["remaining"]),
                "X-RateLimit-Reset": str(minute_info["reset"]),
                "Retry-After": str(minute_info["retry_after"]),
            },
        )

    # Check daily quota
    daily_allowed, daily_info = await rate_limiter.check_daily_quota(
        identifier=org.id,
        daily_limit=limits["requests_per_day"],
    )

    if not daily_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "DAILY_QUOTA_EXCEEDED",
                "message": f"Daily quota of {limits['requests_per_day']} requests exceeded",
            },
            headers={
                "X-DailyLimit-Limit": str(daily_info["daily_limit"]),
                "X-DailyLimit-Remaining": "0",
            },
        )

    # Check concurrent limit
    concurrent_allowed, concurrent_info = await rate_limiter.check_concurrent(
        identifier=org.id,
        max_concurrent=limits["concurrent"],
        request_id=request_id,
    )

    if not concurrent_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "CONCURRENT_LIMIT_EXCEEDED",
                "message": f"Maximum concurrent requests ({limits['concurrent']}) exceeded",
            },
        )

    # Store for cleanup and response headers
    request.state.rate_limit_info = {
        **minute_info,
        **daily_info,
        **concurrent_info,
        "request_id": request_id,
    }

    return request.state.rate_limit_info


class RateLimitMiddleware:
    """
    ASGI middleware that adds rate limit headers to all responses.
    The actual checking is done via FastAPI dependencies.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Add rate limit headers if available
                headers = list(message.get("headers", []))
                state = scope.get("state", {})

                if "rate_limit_info" in state:
                    info = state["rate_limit_info"]
                    headers.extend([
                        (b"x-ratelimit-limit", str(info.get("limit", "")).encode()),
                        (b"x-ratelimit-remaining", str(info.get("remaining", "")).encode()),
                        (b"x-ratelimit-reset", str(info.get("reset", "")).encode()),
                    ])

                message["headers"] = headers

            await send(message)

        await self.app(scope, receive, send_wrapper)
