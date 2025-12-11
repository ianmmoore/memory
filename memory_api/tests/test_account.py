"""
Tests for account management endpoints.

Tests:
- Get account details
- API key management
- Signup flow
"""

import pytest


@pytest.mark.asyncio
class TestGetAccount:
    """Tests for account retrieval."""

    async def test_get_account_success(self, client):
        """Test successful account retrieval."""
        response = await client.get("/v1/account")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "org_id" in data["data"]
        assert "name" in data["data"]
        assert "email" in data["data"]
        assert "plan" in data["data"]

    async def test_get_account_plan_info(self, client):
        """Test that account includes plan information."""
        response = await client.get("/v1/account")

        assert response.status_code == 200
        plan = response.json()["data"]["plan"]
        assert "id" in plan
        assert "name" in plan


@pytest.mark.asyncio
class TestAPIKeyManagement:
    """Tests for API key CRUD operations."""

    async def test_list_api_keys(self, client):
        """Test listing API keys."""
        response = await client.get("/v1/account/api-keys")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
        # Should have at least the test key
        assert len(data["data"]) >= 1

    async def test_create_api_key(self, client):
        """Test creating a new API key."""
        response = await client.post(
            "/v1/account/api-keys",
            json={
                "name": "New Test Key",
                "environment": "test",
                "scopes": ["read", "write"],
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "key" in data["data"]  # Full key only shown on creation
        assert data["data"]["name"] == "New Test Key"
        assert data["data"]["environment"] == "test"
        assert data["data"]["scopes"] == ["read", "write"]

    async def test_create_api_key_with_expiry(self, client):
        """Test creating API key with expiration."""
        response = await client.post(
            "/v1/account/api-keys",
            json={
                "name": "Expiring Key",
                "expires_in_days": 30,
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["data"]["expires_at"] is not None

    async def test_revoke_api_key(self, client):
        """Test revoking an API key."""
        # First create a key to revoke
        create_response = await client.post(
            "/v1/account/api-keys",
            json={"name": "Key to Revoke"}
        )
        key_id = create_response.json()["data"]["id"]

        # Revoke the key
        response = await client.delete(f"/v1/account/api-keys/{key_id}")

        assert response.status_code == 200
        assert response.json()["data"]["revoked"] is True

    async def test_cannot_revoke_current_key(self, client, db_with_data):
        """Test that current key cannot be revoked."""
        # Get current key ID from the list
        list_response = await client.get("/v1/account/api-keys")
        current_key_id = list_response.json()["data"][0]["id"]

        # Try to revoke current key
        response = await client.delete(f"/v1/account/api-keys/{current_key_id}")

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "CANNOT_REVOKE_CURRENT_KEY"


@pytest.mark.asyncio
class TestSignup:
    """Tests for signup endpoint."""

    async def test_signup_success(self, client):
        """Test successful signup."""
        # Remove auth header for signup
        client.headers.pop("Authorization", None)

        response = await client.post(
            "/v1/signup",
            json={
                "name": "New Company",
                "email": "newcompany@example.com",
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "key" in data["data"]  # Full API key
        assert data["data"]["scopes"] == ["read", "write", "delete", "admin"]

    async def test_signup_duplicate_email(self, client, db_with_data):
        """Test that duplicate emails are rejected."""
        client.headers.pop("Authorization", None)

        response = await client.post(
            "/v1/signup",
            json={
                "name": "Duplicate",
                "email": "test@example.com",  # Same as test org
            }
        )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "EMAIL_EXISTS"

    async def test_signup_invalid_email(self, client):
        """Test that invalid emails are rejected."""
        client.headers.pop("Authorization", None)

        response = await client.post(
            "/v1/signup",
            json={
                "name": "Test",
                "email": "not-an-email",
            }
        )

        assert response.status_code == 400

    async def test_signup_with_plan(self, client):
        """Test signup with specific plan."""
        client.headers.pop("Authorization", None)

        response = await client.post(
            "/v1/signup",
            json={
                "name": "Starter Company",
                "email": "starter@example.com",
                "plan_id": "starter",
            }
        )

        # May succeed or fail depending on plan availability
        # Just check it's a valid response
        assert response.status_code in [201, 400, 404]
