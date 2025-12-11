"""
Account management API routes.

Endpoints:
- GET  /account              Get account details
- GET  /account/api-keys     List API keys
- POST /account/api-keys     Create new API key
- DELETE /account/api-keys/{id}  Revoke API key
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import Organization, APIKey, PricingPlan
from api.middleware.auth import (
    CurrentOrg,
    CurrentAPIKey,
    require_scope,
    APIKeyAuth,
)
from api.middleware.rate_limit import check_rate_limit
from api.models.requests import CreateAPIKeyRequest, CreateOrganizationRequest
from api.models.responses import (
    AccountResponse,
    AccountData,
    PlanData,
    APIKeyResponse,
    APIKeyData,
    APIKeyCreatedResponse,
    APIKeyCreatedData,
    APIKeyListResponse,
    SuccessResponse,
    ResponseMeta,
)
from billing.pricing import pricing_service

router = APIRouter(prefix="/account", tags=["Account"])


def build_response_meta(request: Request) -> ResponseMeta:
    """Build standard response metadata."""
    start_time = getattr(request.state, "start_time", time.time())
    return ResponseMeta(
        request_id=getattr(request.state, "request_id", "unknown"),
        processing_time_ms=int((time.time() - start_time) * 1000),
    )


@router.get(
    "",
    response_model=AccountResponse,
    summary="Get account details",
    description="Get current organization account information.",
)
async def get_account(
    request: Request,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Get account details."""
    # Get plan info
    plan = pricing_service.get_plan(org.plan_id)
    plan_data = PlanData(
        id=plan.id if plan else "free",
        name=plan.display_name if plan else "Free",
        description=plan.description if plan else None,
        base_price_cents=plan.base_price_cents if plan else 0,
        billing_period=plan.billing_period if plan else "monthly",
    )

    return AccountResponse(
        data=AccountData(
            org_id=org.id,
            name=org.name,
            email=org.email,
            plan=plan_data,
            is_active=org.is_active,
            created_at=org.created_at,
        ),
        meta=build_response_meta(request),
    )


@router.get(
    "/api-keys",
    response_model=APIKeyListResponse,
    summary="List API keys",
    description="List all API keys for the organization.",
)
async def list_api_keys(
    request: Request,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """List all API keys."""
    result = await db.execute(
        select(APIKey)
        .where(APIKey.org_id == org.id)
        .order_by(APIKey.created_at.desc())
    )
    keys = result.scalars().all()

    return APIKeyListResponse(
        data=[
            APIKeyData(
                id=k.id,
                name=k.name,
                key_prefix=k.key_prefix,
                environment=k.environment,
                scopes=k.scopes,
                is_active=k.is_active,
                created_at=k.created_at,
                expires_at=k.expires_at,
                last_used_at=k.last_used_at,
            )
            for k in keys
        ],
        meta=build_response_meta(request),
    )


@router.post(
    "/api-keys",
    response_model=APIKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description="Create a new API key. The full key is only shown once!",
)
async def create_api_key(
    request: Request,
    body: CreateAPIKeyRequest,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["admin"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key."""
    # Generate key
    full_key, key_hash, key_prefix = APIKeyAuth.generate_key(body.environment)

    # Calculate expiration
    expires_at = None
    if body.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)

    # Create key record
    new_key = APIKey(
        org_id=org.id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=body.name,
        environment=body.environment,
        scopes=body.scopes,
        expires_at=expires_at,
    )
    db.add(new_key)
    await db.flush()
    await db.refresh(new_key)

    return APIKeyCreatedResponse(
        data=APIKeyCreatedData(
            id=new_key.id,
            name=new_key.name,
            key=full_key,  # Only shown once!
            key_prefix=new_key.key_prefix,
            environment=new_key.environment,
            scopes=new_key.scopes,
            created_at=new_key.created_at,
            expires_at=new_key.expires_at,
        ),
        meta=build_response_meta(request),
    )


@router.delete(
    "/api-keys/{key_id}",
    response_model=SuccessResponse,
    summary="Revoke API key",
    description="Revoke (disable) an API key.",
)
async def revoke_api_key(
    request: Request,
    key_id: str,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["admin"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Revoke an API key."""
    result = await db.execute(
        select(APIKey).where(
            APIKey.org_id == org.id,
            APIKey.id == key_id,
        )
    )
    key = result.scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "API_KEY_NOT_FOUND",
                "message": f"API key '{key_id}' not found",
            },
        )

    # Cannot revoke the key being used for this request
    if key.id == api_key.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "CANNOT_REVOKE_CURRENT_KEY",
                "message": "Cannot revoke the API key currently in use",
            },
        )

    key.is_active = False
    await db.flush()

    return SuccessResponse(
        data={"revoked": True, "key_id": key_id},
        meta=build_response_meta(request),
    )


# =============================================================================
# Signup endpoint (unauthenticated)
# =============================================================================

signup_router = APIRouter(tags=["Account"])


@signup_router.post(
    "/signup",
    response_model=APIKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new account",
    description="Sign up for a new account. Returns an API key.",
)
async def signup(
    request: Request,
    body: CreateOrganizationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new organization and return an API key."""
    # Check if email already exists
    result = await db.execute(
        select(Organization).where(Organization.email == body.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "EMAIL_EXISTS",
                "message": "An account with this email already exists",
            },
        )

    # Create organization
    org = Organization(
        name=body.name,
        email=body.email,
        plan_id=body.plan_id or "free",
    )
    db.add(org)
    await db.flush()

    # Generate initial API key
    full_key, key_hash, key_prefix = APIKeyAuth.generate_key("live")

    api_key = APIKey(
        org_id=org.id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name="Default Key",
        environment="live",
        scopes=["read", "write", "delete", "admin"],
    )
    db.add(api_key)
    await db.flush()
    await db.refresh(api_key)

    return APIKeyCreatedResponse(
        data=APIKeyCreatedData(
            id=api_key.id,
            name=api_key.name,
            key=full_key,
            key_prefix=api_key.key_prefix,
            environment=api_key.environment,
            scopes=api_key.scopes,
            created_at=api_key.created_at,
            expires_at=api_key.expires_at,
        ),
        meta=build_response_meta(request),
    )
