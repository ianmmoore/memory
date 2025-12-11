"""
Pytest configuration and fixtures for Memory API tests.

Provides:
- Async test client
- Test database setup/teardown
- Mock Redis
- Test organization and API keys
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.models import Base, Organization, APIKey, PricingPlan
from db.database import get_db
from api.main import create_app
from api.middleware.auth import APIKeyAuth
from config.settings import Settings


# =============================================================================
# Test Settings
# =============================================================================

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Override settings for testing."""
    return Settings(
        environment="testing",
        debug=True,
        database_url="sqlite+aiosqlite:///:memory:",
        redis_url="redis://localhost:6379/15",  # Use different DB for tests
        secret_key="test-secret-key-do-not-use-in-production",
        openai_api_key="test-key",
    )


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create async engine for tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for tests."""
    async_session = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="function")
async def db_with_data(db_session: AsyncSession) -> AsyncSession:
    """Database session with test data."""
    # Create default pricing plan
    plan = PricingPlan(
        id="free",
        version="2024-01",
        display_name="Free",
        description="Free tier",
        base_price_cents=0,
        billing_period="monthly",
        config={
            "limits": {
                "api_calls": 100,
                "tokens_processed": 10000,
                "memories_stored": 100,
            },
            "rate_limits": {
                "requests_per_minute": 10,
            },
        },
        is_active=True,
        is_public=True,
        effective_from=datetime.now(timezone.utc).date(),
    )
    db_session.add(plan)

    # Create test organization
    org = Organization(
        id="org_test123",
        name="Test Organization",
        email="test@example.com",
        plan_id="free",
        is_active=True,
    )
    db_session.add(org)

    # Create test API key
    full_key, key_hash, key_prefix = APIKeyAuth.generate_key("test")
    api_key = APIKey(
        id="key_test123",
        org_id="org_test123",
        key_hash=key_hash,
        key_prefix=key_prefix,
        name="Test Key",
        environment="test",
        scopes=["read", "write", "delete", "admin"],
        is_active=True,
    )
    db_session.add(api_key)

    await db_session.commit()

    # Store the full key for tests to use
    db_session.info["test_api_key"] = full_key
    db_session.info["test_org_id"] = org.id

    yield db_session


# =============================================================================
# Redis Mock
# =============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client for tests."""
    mock = AsyncMock()

    # Mock common Redis operations
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.incr = AsyncMock(return_value=1)
    mock.expire = AsyncMock(return_value=True)
    mock.lpush = AsyncMock(return_value=1)
    mock.rpop = AsyncMock(return_value=None)
    mock.hgetall = AsyncMock(return_value={})
    mock.hincrby = AsyncMock(return_value=1)
    mock.eval = AsyncMock(return_value=[1, 1, 100])  # Rate limit allowed
    mock.pipeline = MagicMock(return_value=mock)
    mock.execute = AsyncMock(return_value=[])
    mock.sadd = AsyncMock(return_value=1)
    mock.scard = AsyncMock(return_value=1)
    mock.srem = AsyncMock(return_value=1)
    mock.ping = AsyncMock(return_value=True)
    mock.close = AsyncMock()

    return mock


# =============================================================================
# FastAPI Test Client
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def app(db_with_data: AsyncSession, mock_redis):
    """Create FastAPI app with test dependencies."""
    from api.middleware.rate_limit import rate_limiter
    from api.middleware.usage_tracking import usage_tracker

    # Create app
    application = create_app()

    # Override database dependency
    async def override_get_db():
        yield db_with_data

    application.dependency_overrides[get_db] = override_get_db

    # Mock Redis connections
    rate_limiter._redis = mock_redis
    usage_tracker._redis = mock_redis

    yield application

    # Cleanup
    application.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def client(app, db_with_data: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Add API key to client headers
        api_key = db_with_data.info.get("test_api_key")
        if api_key:
            ac.headers["Authorization"] = f"Bearer {api_key}"
        yield ac


@pytest.fixture
def sync_client(app) -> Generator[TestClient, None, None]:
    """Synchronous test client for simple tests."""
    with TestClient(app) as c:
        yield c


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_memory():
    """Sample memory data for tests."""
    return {
        "text": "User prefers dark mode interfaces",
        "metadata": {
            "category": "preference",
            "confidence": 0.95,
        },
    }


@pytest.fixture
def sample_memories():
    """Multiple sample memories for batch tests."""
    return [
        {
            "text": "User works as a software engineer",
            "metadata": {"category": "professional"},
        },
        {
            "text": "User prefers Python over JavaScript",
            "metadata": {"category": "preference"},
        },
        {
            "text": "User lives in San Francisco",
            "metadata": {"category": "location"},
        },
    ]


@pytest.fixture
def sample_query():
    """Sample query for retrieval tests."""
    return {
        "context": "What programming language does the user prefer?",
        "max_memories": 10,
        "relevance_threshold": 0.3,
    }


# =============================================================================
# Event Loop Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
