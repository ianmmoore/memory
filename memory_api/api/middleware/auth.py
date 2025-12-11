"""
Authentication middleware for API key validation.

Security features:
- API keys are hashed with bcrypt before storage
- Only key prefix is stored in plaintext for identification
- Keys can have scopes (read, write, delete, admin)
- Keys can have expiration dates
- Last usage time is tracked
"""

import secrets
import hashlib
from datetime import datetime, timezone
from typing import Optional, Annotated
from functools import wraps

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from db.database import get_db
from db.models import APIKey, Organization


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer scheme for API key extraction
bearer_scheme = HTTPBearer(
    scheme_name="API Key",
    description="API key in format: mem_live_xxx or mem_test_xxx",
    auto_error=False,
)


class APIKeyAuth:
    """
    API Key authentication handler.

    Handles creation, validation, and management of API keys.
    Keys follow the format: {prefix}_{environment}_{random}
    Example: mem_live_a1b2c3d4e5f6...
    """

    @staticmethod
    def generate_key(environment: str = "live") -> tuple[str, str, str]:
        """
        Generate a new API key.

        Returns:
            tuple: (full_key, key_hash, key_prefix)
        """
        prefix = settings.api_key_prefix
        random_part = secrets.token_hex(settings.api_key_length)
        full_key = f"{prefix}_{environment}_{random_part}"

        # Hash the full key for storage
        key_hash = pwd_context.hash(full_key)

        # Store prefix for identification (first 12 chars of random part)
        key_prefix = f"{prefix}_{environment}_{random_part[:12]}"

        return full_key, key_hash, key_prefix

    @staticmethod
    def verify_key(plain_key: str, hashed_key: str) -> bool:
        """Verify an API key against its hash."""
        return pwd_context.verify(plain_key, hashed_key)

    @staticmethod
    def get_key_prefix(full_key: str) -> str:
        """Extract the prefix from a full API key for lookup."""
        parts = full_key.split("_")
        if len(parts) >= 3:
            # mem_live_randompart -> mem_live_first12chars
            return f"{parts[0]}_{parts[1]}_{parts[2][:12]}"
        return full_key[:20]

    @staticmethod
    def hash_for_lookup(full_key: str) -> str:
        """
        Create a fast lookup hash (not for security, just for indexing).
        The actual verification still uses bcrypt.
        """
        return hashlib.sha256(full_key.encode()).hexdigest()


async def get_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[APIKey]:
    """
    Extract and validate API key from request.

    Looks for API key in:
    1. Authorization header (Bearer token)
    2. X-API-Key header (fallback)

    Returns None if no key provided (for public endpoints).
    Raises HTTPException if key is invalid.
    """
    # Try Authorization header first
    api_key_str = None
    if credentials:
        api_key_str = credentials.credentials
    else:
        # Fallback to X-API-Key header
        api_key_str = request.headers.get("X-API-Key")

    if not api_key_str:
        return None

    # Validate key format
    if not api_key_str.startswith(settings.api_key_prefix):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_API_KEY_FORMAT",
                "message": f"API key must start with '{settings.api_key_prefix}_'",
            },
        )

    # Look up key by prefix
    key_prefix = APIKeyAuth.get_key_prefix(api_key_str)

    result = await db.execute(
        select(APIKey).where(APIKey.key_prefix == key_prefix)
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_API_KEY",
                "message": "API key not found",
            },
        )

    # Verify the full key
    if not APIKeyAuth.verify_key(api_key_str, api_key.key_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_API_KEY",
                "message": "API key verification failed",
            },
        )

    # Check if key is active
    if not api_key.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "API_KEY_DISABLED",
                "message": "This API key has been disabled",
            },
        )

    # Check expiration
    if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "API_KEY_EXPIRED",
                "message": "This API key has expired",
            },
        )

    # Update last used timestamp (async, don't wait)
    await db.execute(
        update(APIKey)
        .where(APIKey.id == api_key.id)
        .values(last_used_at=datetime.now(timezone.utc))
    )

    # Store key info in request state for later use
    request.state.api_key = api_key
    request.state.org_id = api_key.org_id

    return api_key


async def verify_api_key(
    api_key: Optional[APIKey] = Depends(get_api_key),
) -> APIKey:
    """
    Dependency that requires a valid API key.
    Use this for protected endpoints.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "MISSING_API_KEY",
                "message": "API key is required. Provide via Authorization header or X-API-Key header.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    return api_key


async def get_current_org(
    api_key: APIKey = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> Organization:
    """
    Get the organization associated with the current API key.
    """
    result = await db.execute(
        select(Organization).where(Organization.id == api_key.org_id)
    )
    org = result.scalar_one_or_none()

    if not org:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "ORGANIZATION_NOT_FOUND",
                "message": "Organization associated with this API key not found",
            },
        )

    if not org.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "ORGANIZATION_DISABLED",
                "message": "This organization has been disabled",
            },
        )

    return org


def require_scope(required_scopes: list[str]):
    """
    Dependency factory that checks if the API key has required scopes.

    Usage:
        @app.get("/admin/users")
        async def admin_endpoint(
            _: None = Depends(require_scope(["admin"]))
        ):
            ...
    """
    async def scope_checker(api_key: APIKey = Depends(verify_api_key)):
        key_scopes = set(api_key.scopes or [])

        # Admin scope grants all permissions
        if "admin" in key_scopes:
            return api_key

        # Check if key has all required scopes
        if not all(scope in key_scopes for scope in required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "INSUFFICIENT_PERMISSIONS",
                    "message": f"This operation requires scopes: {required_scopes}",
                    "required_scopes": required_scopes,
                    "your_scopes": list(key_scopes),
                },
            )

        return api_key

    return scope_checker


class AuthMiddleware:
    """
    ASGI middleware for authentication.
    Adds org_id and api_key_id to request state for all requests.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Initialize state
            scope.setdefault("state", {})
            scope["state"]["org_id"] = None
            scope["state"]["api_key_id"] = None

        await self.app(scope, receive, send)


# Type aliases for dependency injection
CurrentOrg = Annotated[Organization, Depends(get_current_org)]
CurrentAPIKey = Annotated[APIKey, Depends(verify_api_key)]
OptionalAPIKey = Annotated[Optional[APIKey], Depends(get_api_key)]
