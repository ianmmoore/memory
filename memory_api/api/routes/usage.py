"""
Usage and billing API routes.

Endpoints:
- GET /usage              Get current usage
- GET /usage/history      Get historical usage
- GET /usage/breakdown    Get usage by endpoint
- GET /usage/projection   Get projected usage
- GET /usage/export       Export usage data
"""

import time
from datetime import datetime, timezone, timedelta, date
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import Organization
from api.middleware.auth import CurrentOrg, CurrentAPIKey, require_scope
from api.middleware.rate_limit import check_rate_limit
from api.models.responses import (
    UsageResponse,
    UsageResponseData,
    UsageData,
    UsageLimits,
    SuccessResponse,
    ResponseMeta,
)
from billing.pricing import pricing_service
from billing.usage import usage_service

router = APIRouter(prefix="/usage", tags=["Usage & Billing"])


def build_response_meta(request: Request) -> ResponseMeta:
    """Build standard response metadata."""
    start_time = getattr(request.state, "start_time", time.time())
    return ResponseMeta(
        request_id=getattr(request.state, "request_id", "unknown"),
        processing_time_ms=int((time.time() - start_time) * 1000),
    )


@router.get(
    "",
    response_model=UsageResponse,
    summary="Get current usage",
    description="Get usage for the current billing period with limits.",
)
async def get_usage(
    request: Request,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Get current period usage."""
    # Get current usage
    current = await usage_service.get_current_period_usage(org.id, db)

    # Get plan limits
    pricing = await pricing_service.get_effective_pricing(org.id, db)
    limits = pricing.limits

    # Build limits response
    limits_data = UsageLimits(
        api_calls=limits.api_calls,
        tokens_processed=limits.tokens_processed,
        memories_stored=limits.memories_stored,
        embeddings_generated=limits.embeddings_generated,
    )

    # Calculate percent used
    def calc_percent(used: int, limit: Optional[int]) -> float:
        if limit is None or limit == 0:
            return 0.0
        return min(100.0, (used / limit) * 100)

    percent_used = {
        "api_calls": calc_percent(current.api_calls, limits.api_calls),
        "tokens_processed": calc_percent(current.tokens_total, limits.tokens_processed),
        "memories_stored": calc_percent(current.memories_stored, limits.memories_stored),
        "embeddings_generated": calc_percent(current.embeddings_generated, limits.embeddings_generated),
    }

    return UsageResponse(
        data=UsageResponseData(
            current=UsageData(
                period=current.period,
                api_calls=current.api_calls,
                tokens_processed=current.tokens_total,
                tokens_input=current.tokens_input,
                tokens_output=current.tokens_output,
                memories_stored=current.memories_stored,
                memories_read=current.memories_read,
                memories_written=current.memories_written,
                embeddings_generated=current.embeddings_generated,
                estimated_cost_cents=current.cost_cents,
            ),
            limits=limits_data,
            percent_used=percent_used,
        ),
        meta=build_response_meta(request),
    )


@router.get(
    "/history",
    response_model=SuccessResponse,
    summary="Get usage history",
    description="Get historical daily usage data.",
)
async def get_usage_history(
    request: Request,
    days: int = Query(default=30, ge=1, le=90, description="Number of days of history"),
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Get historical usage."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    history = await usage_service.get_usage_history(org.id, start_date, end_date, db)

    return SuccessResponse(
        data={
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "history": [
                {
                    "date": h.date.isoformat(),
                    "api_calls": h.api_calls,
                    "tokens_total": h.tokens_total,
                    "memories_read": h.memories_read,
                    "memories_written": h.memories_written,
                    "cost_cents": h.cost_cents,
                }
                for h in history
            ],
        },
        meta=build_response_meta(request),
    )


@router.get(
    "/breakdown",
    response_model=SuccessResponse,
    summary="Get usage breakdown",
    description="Get usage broken down by endpoint.",
)
async def get_usage_breakdown(
    request: Request,
    days: int = Query(default=7, ge=1, le=30, description="Number of days to analyze"),
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Get usage breakdown by endpoint."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    breakdown = await usage_service.get_usage_by_endpoint(
        org.id, start_date, end_date, db
    )

    return SuccessResponse(
        data={
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "endpoints": breakdown,
        },
        meta=build_response_meta(request),
    )


@router.get(
    "/projection",
    response_model=SuccessResponse,
    summary="Get usage projection",
    description="Get projected usage for the current billing period.",
)
async def get_usage_projection(
    request: Request,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Get projected month-end usage."""
    projection = await usage_service.project_usage(org.id, db)

    return SuccessResponse(
        data=projection,
        meta=build_response_meta(request),
    )


@router.get(
    "/export",
    summary="Export usage data",
    description="Export detailed usage data in CSV or JSON format.",
)
async def export_usage(
    request: Request,
    format: str = Query(default="csv", description="Export format: csv or json"),
    days: int = Query(default=30, ge=1, le=90, description="Number of days to export"),
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Export usage data."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    data = await usage_service.export_usage(
        org.id, start_date, end_date, format, db
    )

    if format == "json":
        return Response(
            content=data,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=usage_{org.id}_{start_date.date()}_{end_date.date()}.json"
            },
        )
    else:
        return Response(
            content=data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=usage_{org.id}_{start_date.date()}_{end_date.date()}.csv"
            },
        )
