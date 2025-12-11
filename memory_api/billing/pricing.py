"""
Pricing engine for billing calculations.

Features:
- Configuration-driven pricing from YAML
- Per-customer pricing overrides
- Volume discounts
- Promotional credits
- Plan grandfathering
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Any
from pathlib import Path
from decimal import Decimal

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from db.models import (
    Organization,
    PricingPlan,
    CustomerPricing,
    CustomerAddon,
    CustomerCredit,
)


@dataclass
class PlanLimits:
    """Plan usage limits."""
    api_calls: Optional[int] = None
    tokens_processed: Optional[int] = None
    memories_stored: Optional[int] = None
    embeddings_generated: Optional[int] = None
    queries_executed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "PlanLimits":
        if not data:
            return cls()
        return cls(
            api_calls=data.get("api_calls"),
            tokens_processed=data.get("tokens_processed"),
            memories_stored=data.get("memories_stored"),
            embeddings_generated=data.get("embeddings_generated"),
            queries_executed=data.get("queries_executed"),
        )


@dataclass
class OverageRate:
    """Overage pricing for a metric."""
    price_cents: int
    per_units: int


@dataclass
class PlanConfig:
    """Full plan configuration."""
    id: str
    display_name: str
    description: str
    base_price_cents: Optional[int]
    billing_period: str
    limits: PlanLimits
    rate_limits: dict
    features: list[str]
    overage_allowed: bool
    overage_rates: dict[str, OverageRate]
    trial_days: int = 0


@dataclass
class EffectivePricing:
    """Effective pricing for a customer (including overrides)."""
    plan: PlanConfig
    limits: PlanLimits
    overage_rates: dict[str, OverageRate]
    discount_percent: float = 0.0
    addons: list[dict] = field(default_factory=list)
    available_credits_cents: int = 0


@dataclass
class LineItem:
    """Invoice line item."""
    description: str
    amount_cents: int
    quantity: Optional[int] = None
    unit_price_cents: Optional[int] = None
    discount_percent: float = 0.0


@dataclass
class Invoice:
    """Generated invoice."""
    org_id: str
    period_start: date
    period_end: date
    line_items: list[LineItem]
    subtotal_cents: int
    discount_cents: int
    credits_applied_cents: int
    total_cents: int


class PricingService:
    """
    Configuration-driven pricing engine.

    Loads pricing configuration from YAML and database,
    calculates effective pricing per customer, and generates invoices.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(
            Path(__file__).parent.parent / "config" / "pricing_config.yaml"
        )
        self._config: Optional[dict] = None
        self._plans: dict[str, PlanConfig] = {}

    def _load_config(self) -> dict:
        """Load pricing configuration from YAML."""
        if self._config is None:
            with open(self.config_path, "r") as f:
                self._config = yaml.safe_load(f)
            self._parse_plans()
        return self._config

    def _parse_plans(self):
        """Parse plan configurations."""
        config = self._load_config()
        for plan_id, plan_data in config.get("plans", {}).items():
            limits = PlanLimits.from_dict(plan_data.get("limits"))

            overage_rates = {}
            if plan_data.get("overage_rates"):
                for metric, rate_data in plan_data["overage_rates"].items():
                    overage_rates[metric] = OverageRate(
                        price_cents=rate_data["price_cents"],
                        per_units=rate_data["per_units"],
                    )

            self._plans[plan_id] = PlanConfig(
                id=plan_id,
                display_name=plan_data.get("display_name", plan_id),
                description=plan_data.get("description", ""),
                base_price_cents=plan_data.get("base_price_cents"),
                billing_period=plan_data.get("billing_period", "monthly"),
                limits=limits,
                rate_limits=plan_data.get("rate_limits", {}),
                features=plan_data.get("features", []),
                overage_allowed=plan_data.get("overage_allowed", False),
                overage_rates=overage_rates,
                trial_days=plan_data.get("trial_days", 0),
            )

    @property
    def config(self) -> dict:
        return self._load_config()

    def get_plan(self, plan_id: str) -> Optional[PlanConfig]:
        """Get plan configuration by ID."""
        self._load_config()
        return self._plans.get(plan_id)

    def list_plans(self, public_only: bool = True) -> list[PlanConfig]:
        """List available plans."""
        self._load_config()
        plans = list(self._plans.values())
        if public_only:
            plans = [p for p in plans if p.id != "enterprise"]
        return plans

    async def get_effective_pricing(
        self,
        org_id: str,
        db: AsyncSession,
    ) -> EffectivePricing:
        """
        Get effective pricing for an organization.

        Combines:
        - Base plan pricing
        - Custom pricing overrides
        - Active add-ons
        - Available credits
        """
        # Get organization
        result = await db.execute(
            select(Organization).where(Organization.id == org_id)
        )
        org = result.scalar_one_or_none()
        if not org:
            raise ValueError(f"Organization {org_id} not found")

        # Get base plan
        plan = self.get_plan(org.plan_id)
        if not plan:
            plan = self.get_plan("free")

        # Get custom pricing overrides
        result = await db.execute(
            select(CustomerPricing).where(CustomerPricing.org_id == org_id)
        )
        custom_pricing = result.scalar_one_or_none()

        # Merge limits with custom overrides
        limits = plan.limits
        overage_rates = plan.overage_rates.copy()
        discount_percent = 0.0

        if custom_pricing:
            if custom_pricing.custom_limits:
                limits = PlanLimits(
                    api_calls=custom_pricing.custom_limits.get("api_calls", limits.api_calls),
                    tokens_processed=custom_pricing.custom_limits.get("tokens_processed", limits.tokens_processed),
                    memories_stored=custom_pricing.custom_limits.get("memories_stored", limits.memories_stored),
                    embeddings_generated=custom_pricing.custom_limits.get("embeddings_generated", limits.embeddings_generated),
                    queries_executed=custom_pricing.custom_limits.get("queries_executed", limits.queries_executed),
                )

            if custom_pricing.custom_rates:
                for metric, rate_data in custom_pricing.custom_rates.items():
                    overage_rates[metric] = OverageRate(
                        price_cents=rate_data["price_cents"],
                        per_units=rate_data["per_units"],
                    )

            if custom_pricing.discount_percent:
                discount_percent = float(custom_pricing.discount_percent)

        # Get active add-ons
        result = await db.execute(
            select(CustomerAddon).where(
                CustomerAddon.org_id == org_id,
                CustomerAddon.is_active == True,
            )
        )
        addons = [
            {"id": a.addon_id, "quantity": a.quantity}
            for a in result.scalars().all()
        ]

        # Get available credits
        result = await db.execute(
            select(CustomerCredit).where(
                CustomerCredit.org_id == org_id,
                CustomerCredit.remaining_cents > 0,
            )
        )
        credits = result.scalars().all()
        total_credits = sum(
            c.remaining_cents
            for c in credits
            if not c.expires_at or c.expires_at >= date.today()
        )

        return EffectivePricing(
            plan=plan,
            limits=limits,
            overage_rates=overage_rates,
            discount_percent=discount_percent,
            addons=addons,
            available_credits_cents=total_credits,
        )

    def _get_volume_discount(self, metric: str, amount: int) -> float:
        """Get volume discount percentage for a metric."""
        discounts = self.config.get("volume_discounts", {}).get(metric, [])
        discount = 0.0
        for tier in discounts:
            if amount >= tier["threshold"]:
                discount = tier["discount_percent"] / 100.0
        return discount

    async def calculate_invoice(
        self,
        org_id: str,
        usage: dict,
        period_start: date,
        period_end: date,
        db: AsyncSession,
    ) -> Invoice:
        """
        Calculate invoice for a billing period.

        Args:
            org_id: Organization ID
            usage: Usage data dict with metric totals
            period_start: Billing period start
            period_end: Billing period end
            db: Database session

        Returns:
            Invoice with line items
        """
        pricing = await self.get_effective_pricing(org_id, db)
        line_items = []

        # Base subscription
        if pricing.plan.base_price_cents:
            line_items.append(LineItem(
                description=f"{pricing.plan.display_name} Plan - {pricing.plan.billing_period}",
                amount_cents=pricing.plan.base_price_cents,
            ))

        # Overage charges
        if pricing.plan.overage_allowed:
            for metric, rate in pricing.overage_rates.items():
                used = usage.get(metric, 0)
                limit = getattr(pricing.limits, metric, None)

                if limit is not None and used > limit:
                    overage = used - limit
                    units = overage / rate.per_units
                    base_charge = int(units * rate.price_cents)

                    # Apply volume discount
                    discount = self._get_volume_discount(metric, used)
                    charge = int(base_charge * (1 - discount))

                    line_items.append(LineItem(
                        description=f"{metric.replace('_', ' ').title()} overage ({overage:,} units)",
                        amount_cents=charge,
                        quantity=overage,
                        unit_price_cents=rate.price_cents,
                        discount_percent=discount * 100,
                    ))

        # Add-ons
        addons_config = self.config.get("addons", {})
        for addon in pricing.addons:
            addon_config = addons_config.get(addon["id"], {})
            price = addon_config.get("price_cents", 0) * addon.get("quantity", 1)
            line_items.append(LineItem(
                description=addon_config.get("display_name", addon["id"]),
                amount_cents=price,
                quantity=addon.get("quantity", 1),
            ))

        # Calculate totals
        subtotal = sum(item.amount_cents for item in line_items)

        # Apply customer discount
        discount_cents = int(subtotal * (pricing.discount_percent / 100))

        # Apply credits
        after_discount = subtotal - discount_cents
        credits_applied = min(pricing.available_credits_cents, after_discount)

        total = after_discount - credits_applied

        return Invoice(
            org_id=org_id,
            period_start=period_start,
            period_end=period_end,
            line_items=line_items,
            subtotal_cents=subtotal,
            discount_cents=discount_cents,
            credits_applied_cents=credits_applied,
            total_cents=max(0, total),
        )

    async def check_quota(
        self,
        org_id: str,
        metric: str,
        db: AsyncSession,
    ) -> dict:
        """
        Check if organization is within quota for a metric.

        Returns dict with:
        - allowed: bool
        - limit: int or None
        - used: int
        - remaining: int or None
        """
        pricing = await self.get_effective_pricing(org_id, db)
        limit = getattr(pricing.limits, metric, None)

        # Get current usage from Redis (real-time counter)
        from api.middleware.usage_tracking import usage_tracker
        current_usage = await usage_tracker.get_current_usage(org_id)
        used = current_usage.get(metric, 0)

        if limit is None:
            return {
                "allowed": True,
                "limit": None,
                "used": used,
                "remaining": None,
            }

        remaining = max(0, limit - used)
        allowed = used < limit or pricing.plan.overage_allowed

        return {
            "allowed": allowed,
            "limit": limit,
            "used": used,
            "remaining": remaining,
        }


# Global instance
pricing_service = PricingService()
