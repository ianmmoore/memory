"""
Health check and status endpoints.

Endpoints:
- GET /health     Health check for load balancers
- GET /          Root endpoint with API info
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from redis import asyncio as aioredis

from db.database import get_db
from config.settings import settings
from api.models.responses import HealthResponse, HealthData

router = APIRouter(tags=["Health"])


async def check_database(db: AsyncSession) -> str:
    """Check database connectivity."""
    try:
        await db.execute(text("SELECT 1"))
        return "healthy"
    except Exception as e:
        return f"unhealthy: {str(e)}"


async def check_redis() -> str:
    """Check Redis connectivity."""
    try:
        redis = await aioredis.from_url(settings.redis_url)
        await redis.ping()
        await redis.close()
        return "healthy"
    except Exception as e:
        return f"unhealthy: {str(e)}"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Health check endpoint for load balancers and monitoring.",
)
async def health_check(
    db: AsyncSession = Depends(get_db),
):
    """
    Health check endpoint.

    Returns status of all dependencies.
    Used by load balancers to determine instance health.
    """
    db_status = await check_database(db)
    redis_status = await check_redis()

    # Overall status
    all_healthy = db_status == "healthy" and redis_status == "healthy"
    status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        success=all_healthy,
        data=HealthData(
            status=status,
            version=settings.app_version,
            database=db_status,
            redis=redis_status,
        ),
    )


@router.get(
    "/",
    summary="API root",
    description="Root endpoint with API information.",
)
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "documentation": "/docs",
        "openapi": "/openapi.json",
    }
