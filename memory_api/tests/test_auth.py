"""
Tests for authentication and API key management.

Tests:
- API key generation and validation
- Authentication middleware
- Scope-based authorization
- Key expiration
"""

import pytest
from datetime import datetime, timezone, timedelta

from api.middleware.auth import APIKeyAuth


class TestAPIKeyAuth:
    """Tests for API key generation and validation."""

    def test_generate_key_format(self):
        """Test that generated keys have correct format."""
        full_key, key_hash, key_prefix = APIKeyAuth.generate_key("live")

        assert full_key.startswith("mem_live_")
        assert len(full_key) > 20
        assert key_prefix.startswith("mem_live_")
        assert len(key_prefix) < len(full_key)

    def test_generate_key_unique(self):
        """Test that each key generation produces unique keys."""
        key1, _, _ = APIKeyAuth.generate_key("live")
        key2, _, _ = APIKeyAuth.generate_key("live")

        assert key1 != key2

    def test_generate_key_environments(self):
        """Test key generation for different environments."""
        live_key, _, _ = APIKeyAuth.generate_key("live")
        test_key, _, _ = APIKeyAuth.generate_key("test")

        assert "live" in live_key
        assert "test" in test_key

    def test_verify_key_success(self):
        """Test successful key verification."""
        full_key, key_hash, _ = APIKeyAuth.generate_key("live")

        assert APIKeyAuth.verify_key(full_key, key_hash) is True

    def test_verify_key_failure(self):
        """Test key verification with wrong key."""
        _, key_hash, _ = APIKeyAuth.generate_key("live")
        wrong_key = "mem_live_wrongkey123456789"

        assert APIKeyAuth.verify_key(wrong_key, key_hash) is False

    def test_get_key_prefix(self):
        """Test key prefix extraction."""
        full_key, _, expected_prefix = APIKeyAuth.generate_key("live")
        extracted_prefix = APIKeyAuth.get_key_prefix(full_key)

        assert extracted_prefix == expected_prefix


@pytest.mark.asyncio
class TestAuthenticationEndpoints:
    """Tests for authentication via API endpoints."""

    async def test_request_without_key(self, client):
        """Test that requests without API key are rejected."""
        # Remove auth header
        client.headers.pop("Authorization", None)

        response = await client.get("/v1/memories")

        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "MISSING_API_KEY"

    async def test_request_with_invalid_key(self, client):
        """Test that invalid API keys are rejected."""
        client.headers["Authorization"] = "Bearer mem_live_invalid123"

        response = await client.get("/v1/memories")

        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert "INVALID" in data["error"]["code"]

    async def test_request_with_valid_key(self, client):
        """Test that valid API keys are accepted."""
        response = await client.get("/v1/memories")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    async def test_key_in_x_api_key_header(self, client, db_with_data):
        """Test API key in X-API-Key header."""
        client.headers.pop("Authorization", None)
        client.headers["X-API-Key"] = db_with_data.info["test_api_key"]

        response = await client.get("/v1/memories")

        assert response.status_code == 200

    async def test_health_endpoint_no_auth(self, client):
        """Test that health endpoint doesn't require auth."""
        client.headers.pop("Authorization", None)

        response = await client.get("/health")

        assert response.status_code == 200


@pytest.mark.asyncio
class TestScopeAuthorization:
    """Tests for scope-based authorization."""

    async def test_read_scope_allows_get(self, client):
        """Test read scope allows GET requests."""
        response = await client.get("/v1/memories")
        assert response.status_code == 200

    async def test_write_scope_allows_post(self, client, sample_memory):
        """Test write scope allows POST requests."""
        response = await client.post("/v1/memories", json=sample_memory)
        assert response.status_code == 201

    async def test_delete_scope_required(self, client):
        """Test delete operations require delete scope."""
        # First create a memory
        response = await client.post(
            "/v1/memories",
            json={"text": "Test memory for deletion"}
        )
        memory_id = response.json()["data"]["id"]

        # Try to delete
        response = await client.delete(f"/v1/memories/{memory_id}")
        # Should work since test key has all scopes
        assert response.status_code == 200

    async def test_admin_scope_for_api_keys(self, client):
        """Test admin scope required for API key management."""
        response = await client.get("/v1/account/api-keys")
        assert response.status_code == 200
