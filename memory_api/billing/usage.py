"""
Usage service for retrieving and aggregating usage data.

Provides:
- Real-time usage from Redis
- Historical usage from PostgreSQL
- Usage projections
- Export functionality
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import UsageEvent, UsageDailyRollup, Memory, Organization
from api.middleware.usage_tracking import usage_tracker


@dataclass
class UsageSummary:
    """Summary of usage for a period."""
    period: str
    api_calls: int
    tokens_input: int
    tokens_output: int
    tokens_total: int
    memories_stored: int
    memories_read: int
    memories_written: int
    embeddings_generated: int
    cost_cents: int


@dataclass
class UsageHistory:
    """Historical usage data point."""
    date: date
    api_calls: int
    tokens_total: int
    memories_read: int
    memories_written: int
    cost_cents: int


class UsageService:
    """
    Service for retrieving usage data.

    Combines real-time data from Redis with historical data from PostgreSQL.
    """

    async def get_current_period_usage(
        self,
        org_id: str,
        db: AsyncSession,
    ) -> UsageSummary:
        """
        Get usage for the current billing period.

        Uses Redis real-time counters for up-to-date data.
        """
        # Get real-time usage from Redis
        current = await usage_tracker.get_current_usage(org_id)

        # Get current memory count
        result = await db.execute(
            select(func.count(Memory.id)).where(
                Memory.org_id == org_id,
                Memory.is_active == True,
            )
        )
        memory_count = result.scalar_one()

        period = datetime.now(timezone.utc).strftime("%Y-%m")

        return UsageSummary(
            period=period,
            api_calls=current.get("api_calls", 0),
            tokens_input=current.get("tokens_input", 0),
            tokens_output=current.get("tokens_output", 0),
            tokens_total=current.get("tokens_input", 0) + current.get("tokens_output", 0),
            memories_stored=memory_count,
            memories_read=current.get("memories_read", 0),
            memories_written=current.get("memories_written", 0),
            embeddings_generated=current.get("embeddings", 0),
            cost_cents=current.get("cost_cents", 0),
        )

    async def get_usage_history(
        self,
        org_id: str,
        start_date: date,
        end_date: date,
        db: AsyncSession,
    ) -> list[UsageHistory]:
        """
        Get historical usage data from daily rollups.

        Args:
            org_id: Organization ID
            start_date: Start of date range
            end_date: End of date range
            db: Database session

        Returns:
            List of daily usage data points
        """
        result = await db.execute(
            select(UsageDailyRollup)
            .where(
                UsageDailyRollup.org_id == org_id,
                UsageDailyRollup.date >= start_date,
                UsageDailyRollup.date <= end_date,
            )
            .order_by(UsageDailyRollup.date)
        )
        rollups = result.scalars().all()

        return [
            UsageHistory(
                date=r.date,
                api_calls=r.total_requests,
                tokens_total=r.total_tokens_input + r.total_tokens_output,
                memories_read=r.total_memories_read,
                memories_written=r.total_memories_written,
                cost_cents=r.total_cost_cents,
            )
            for r in rollups
        ]

    async def get_usage_by_endpoint(
        self,
        org_id: str,
        start_date: datetime,
        end_date: datetime,
        db: AsyncSession,
    ) -> list[dict]:
        """
        Get usage breakdown by endpoint.

        Returns list of dicts with endpoint stats.
        """
        result = await db.execute(
            select(
                UsageEvent.endpoint,
                func.count(UsageEvent.id).label("count"),
                func.sum(UsageEvent.tokens_input + UsageEvent.tokens_output).label("tokens"),
                func.avg(UsageEvent.processing_time_ms).label("avg_latency"),
            )
            .where(
                UsageEvent.org_id == org_id,
                UsageEvent.timestamp >= start_date,
                UsageEvent.timestamp <= end_date,
            )
            .group_by(UsageEvent.endpoint)
            .order_by(func.count(UsageEvent.id).desc())
        )

        return [
            {
                "endpoint": row.endpoint,
                "request_count": row.count,
                "tokens_total": row.tokens or 0,
                "avg_latency_ms": round(row.avg_latency or 0, 2),
            }
            for row in result.all()
        ]

    async def project_usage(
        self,
        org_id: str,
        db: AsyncSession,
    ) -> dict:
        """
        Project usage for the current billing period.

        Uses linear extrapolation from current usage to estimate month-end totals.
        """
        current = await self.get_current_period_usage(org_id, db)

        # Get days elapsed and remaining in month
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        days_elapsed = max(1, (now - month_start).days + 1)

        # Get days in month
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        days_in_month = (next_month - month_start).days

        # Calculate projection factor
        projection_factor = days_in_month / days_elapsed

        return {
            "period": current.period,
            "days_elapsed": days_elapsed,
            "days_remaining": days_in_month - days_elapsed,
            "current": {
                "api_calls": current.api_calls,
                "tokens_total": current.tokens_total,
                "cost_cents": current.cost_cents,
            },
            "projected": {
                "api_calls": int(current.api_calls * projection_factor),
                "tokens_total": int(current.tokens_total * projection_factor),
                "cost_cents": int(current.cost_cents * projection_factor),
            },
        }

    async def export_usage(
        self,
        org_id: str,
        start_date: datetime,
        end_date: datetime,
        format: str,
        db: AsyncSession,
    ) -> str:
        """
        Export usage data in specified format.

        Args:
            org_id: Organization ID
            start_date: Start of date range
            end_date: End of date range
            format: Export format (csv or json)
            db: Database session

        Returns:
            Exported data as string
        """
        result = await db.execute(
            select(UsageEvent)
            .where(
                UsageEvent.org_id == org_id,
                UsageEvent.timestamp >= start_date,
                UsageEvent.timestamp <= end_date,
            )
            .order_by(UsageEvent.timestamp)
            .limit(10000)  # Limit for safety
        )
        events = result.scalars().all()

        if format == "json":
            import json
            data = [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "endpoint": e.endpoint,
                    "method": e.method,
                    "tokens_input": e.tokens_input,
                    "tokens_output": e.tokens_output,
                    "memories_read": e.memories_read,
                    "memories_written": e.memories_written,
                    "processing_time_ms": e.processing_time_ms,
                    "status_code": e.status_code,
                    "cost_cents": e.cost_cents,
                }
                for e in events
            ]
            return json.dumps(data, indent=2)

        else:  # CSV
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "timestamp", "endpoint", "method", "tokens_input", "tokens_output",
                "memories_read", "memories_written", "processing_time_ms",
                "status_code", "cost_cents"
            ])
            for e in events:
                writer.writerow([
                    e.timestamp.isoformat(),
                    e.endpoint,
                    e.method,
                    e.tokens_input,
                    e.tokens_output,
                    e.memories_read,
                    e.memories_written,
                    e.processing_time_ms,
                    e.status_code,
                    e.cost_cents,
                ])
            return output.getvalue()


# Global instance
usage_service = UsageService()
