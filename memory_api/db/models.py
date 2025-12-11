"""
SQLAlchemy ORM models for the Memory API.

Database schema for:
- Organizations (tenants/customers)
- API Keys (authentication)
- Memories (core data)
- Usage tracking and billing
- Audit logging
"""

import uuid
from datetime import datetime, date
from typing import Optional, Any

from sqlalchemy import (
    Column,
    String,
    Integer,
    BigInteger,
    Boolean,
    DateTime,
    Date,
    Text,
    ForeignKey,
    Index,
    UniqueConstraint,
    Numeric,
    LargeBinary,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.ext.hybrid import hybrid_property


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# =============================================================================
# Organization (Tenant/Customer)
# =============================================================================

class Organization(Base):
    """
    An organization represents a customer/tenant.
    All resources are scoped to an organization for multi-tenancy.
    """
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        default=lambda: f"org_{uuid.uuid4().hex[:16]}"
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    plan_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("pricing_plans.id"),
        default="free"
    )

    # Stripe integration
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(255))

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    api_keys: Mapped[list["APIKey"]] = relationship(back_populates="organization")
    memories: Mapped[list["Memory"]] = relationship(back_populates="organization")
    custom_pricing: Mapped[Optional["CustomerPricing"]] = relationship(back_populates="organization")
    addons: Mapped[list["CustomerAddon"]] = relationship(back_populates="organization")
    credits: Mapped[list["CustomerCredit"]] = relationship(back_populates="organization")

    __table_args__ = (
        Index("ix_organizations_email", "email"),
        Index("ix_organizations_stripe_customer_id", "stripe_customer_id"),
    )


# =============================================================================
# API Keys
# =============================================================================

class APIKey(Base):
    """
    API keys for authentication.
    Keys are stored hashed; only the prefix is visible for identification.
    """
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        default=lambda: f"key_{uuid.uuid4().hex[:16]}"
    )
    org_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False
    )

    # Key storage (hashed)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False)  # e.g., "mem_live_abc..."

    # Metadata
    name: Mapped[str] = mapped_column(String(255), default="Default Key")
    environment: Mapped[str] = mapped_column(String(20), default="live")  # live or test

    # Permissions (scopes)
    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        default=["read", "write"]
    )

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="api_keys")

    __table_args__ = (
        Index("ix_api_keys_org_id", "org_id"),
        Index("ix_api_keys_key_prefix", "key_prefix"),
        Index("ix_api_keys_key_hash", "key_hash"),
    )


# =============================================================================
# Memories (Core Data)
# =============================================================================

class Memory(Base):
    """
    Individual memory entries stored for an organization.
    This is the core data model that powers the memory system.
    """
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        default=lambda: f"mem_{uuid.uuid4().hex[:16]}"
    )
    org_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False
    )

    # Memory content
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict
    )

    # Embedding (stored as binary for efficiency)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100))

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="memories")

    __table_args__ = (
        Index("ix_memories_org_id", "org_id"),
        Index("ix_memories_org_created", "org_id", "created_at"),
        Index("ix_memories_org_active", "org_id", "is_active"),
    )


# =============================================================================
# Usage Tracking
# =============================================================================

class UsageEvent(Base):
    """
    Individual usage events for billing and analytics.
    High-volume table - consider partitioning in production.
    """
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    org_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    api_key_id: Mapped[Optional[str]] = mapped_column(String(50))

    # Event details
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True
    )
    endpoint: Mapped[str] = mapped_column(String(100), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)

    # Usage metrics
    tokens_input: Mapped[int] = mapped_column(Integer, default=0)
    tokens_output: Mapped[int] = mapped_column(Integer, default=0)
    memories_read: Mapped[int] = mapped_column(Integer, default=0)
    memories_written: Mapped[int] = mapped_column(Integer, default=0)
    embeddings_generated: Mapped[int] = mapped_column(Integer, default=0)

    # Performance
    processing_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    status_code: Mapped[int] = mapped_column(Integer, default=200)

    # Cost tracking (internal)
    cost_cents: Mapped[int] = mapped_column(Integer, default=0)

    # Request metadata
    request_id: Mapped[Optional[str]] = mapped_column(String(50))
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(String(500))

    __table_args__ = (
        Index("ix_usage_events_org_timestamp", "org_id", "timestamp"),
        Index("ix_usage_events_billing", "org_id", func.date_trunc("month", "timestamp")),
    )


class UsageDailyRollup(Base):
    """
    Daily aggregated usage for efficient billing queries.
    Populated by a background job from usage_events.
    """
    __tablename__ = "usage_daily_rollups"

    org_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)

    # Aggregated metrics
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens_input: Mapped[int] = mapped_column(BigInteger, default=0)
    total_tokens_output: Mapped[int] = mapped_column(BigInteger, default=0)
    total_memories_read: Mapped[int] = mapped_column(BigInteger, default=0)
    total_memories_written: Mapped[int] = mapped_column(BigInteger, default=0)
    total_embeddings: Mapped[int] = mapped_column(BigInteger, default=0)

    # Cost tracking
    total_cost_cents: Mapped[int] = mapped_column(Integer, default=0)

    # Peak values
    peak_memories_stored: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )


# =============================================================================
# Pricing and Billing
# =============================================================================

class PricingPlan(Base):
    """
    Pricing plan definitions.
    Plans are versioned - old versions kept for grandfathering.
    """
    __tablename__ = "pricing_plans"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Pricing
    base_price_cents: Mapped[Optional[int]] = mapped_column(Integer)  # NULL for custom
    billing_period: Mapped[str] = mapped_column(String(20), default="monthly")

    # Full configuration as JSON
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=True)

    # Validity period
    effective_from: Mapped[date] = mapped_column(Date, nullable=False)
    effective_until: Mapped[Optional[date]] = mapped_column(Date)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("id", "version", name="uq_plan_version"),
    )


class CustomerPricing(Base):
    """
    Customer-specific pricing overrides for enterprise deals.
    """
    __tablename__ = "customer_pricing"

    org_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        primary_key=True
    )
    base_plan_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("pricing_plans.id"),
        nullable=False
    )

    # Custom overrides
    custom_limits: Mapped[Optional[dict]] = mapped_column(JSONB)
    custom_rates: Mapped[Optional[dict]] = mapped_column(JSONB)
    discount_percent: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    negotiated_price_cents: Mapped[Optional[int]] = mapped_column(Integer)

    # Contract period
    contract_start: Mapped[Optional[date]] = mapped_column(Date)
    contract_end: Mapped[Optional[date]] = mapped_column(Date)

    # Notes (internal)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="custom_pricing")


class CustomerAddon(Base):
    """Add-on subscriptions for customers."""
    __tablename__ = "customer_addons"

    org_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        primary_key=True
    )
    addon_id: Mapped[str] = mapped_column(String(50), primary_key=True)

    # Subscription details
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    price_cents_override: Mapped[Optional[int]] = mapped_column(Integer)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    activated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="addons")


class CustomerCredit(Base):
    """Promotional or refund credits for customers."""
    __tablename__ = "customer_credits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    org_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False
    )

    # Credit amounts
    credit_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    remaining_cents: Mapped[int] = mapped_column(Integer, nullable=False)

    # Source tracking
    source: Mapped[str] = mapped_column(String(50), nullable=False)  # promotion, refund, manual
    promotion_code: Mapped[Optional[str]] = mapped_column(String(50))
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Validity
    expires_at: Mapped[Optional[date]] = mapped_column(Date)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="credits")

    __table_args__ = (
        Index("ix_customer_credits_org_id", "org_id"),
    )


# =============================================================================
# Audit Logging
# =============================================================================

class AuditLog(Base):
    """
    Audit log for security-sensitive operations.
    Immutable - no updates or deletes.
    """
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True
    )

    # Who
    org_id: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    api_key_id: Mapped[Optional[str]] = mapped_column(String(50))
    actor_type: Mapped[str] = mapped_column(String(20))  # user, api_key, system

    # What
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[Optional[str]] = mapped_column(String(50))

    # Details
    details: Mapped[Optional[dict]] = mapped_column(JSONB)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(String(500))

    # Result
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("ix_audit_logs_org_timestamp", "org_id", "timestamp"),
        Index("ix_audit_logs_action", "action"),
    )
