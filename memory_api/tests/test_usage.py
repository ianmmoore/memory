"""
Tests for usage tracking and billing endpoints.

Tests:
- Get current usage
- Usage history
- Usage breakdown
- Projections
- Export
"""

import pytest


@pytest.mark.asyncio
class TestGetUsage:
    """Tests for usage retrieval."""

    async def test_get_current_usage(self, client):
        """Test getting current period usage."""
        response = await client.get("/v1/usage")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "current" in data["data"]
        assert "limits" in data["data"]
        assert "percent_used" in data["data"]

    async def test_usage_includes_all_metrics(self, client):
        """Test that usage includes all expected metrics."""
        response = await client.get("/v1/usage")

        current = response.json()["data"]["current"]
        expected_metrics = [
            "api_calls",
            "tokens_processed",
            "memories_stored",
            "memories_read",
            "memories_written",
            "embeddings_generated",
        ]

        for metric in expected_metrics:
            assert metric in current, f"Missing metric: {metric}"

    async def test_usage_limits_based_on_plan(self, client):
        """Test that limits reflect the organization's plan."""
        response = await client.get("/v1/usage")

        limits = response.json()["data"]["limits"]
        # Free plan should have limits
        assert limits["api_calls"] is not None
        assert limits["memories_stored"] is not None


@pytest.mark.asyncio
class TestUsageHistory:
    """Tests for historical usage data."""

    async def test_get_usage_history(self, client):
        """Test getting usage history."""
        response = await client.get("/v1/usage/history")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "history" in data["data"]
        assert isinstance(data["data"]["history"], list)

    async def test_usage_history_custom_days(self, client):
        """Test history with custom date range."""
        response = await client.get("/v1/usage/history?days=7")

        assert response.status_code == 200
        data = response.json()
        assert "period_start" in data["data"]
        assert "period_end" in data["data"]

    async def test_usage_history_max_days(self, client):
        """Test that excessive days are rejected."""
        response = await client.get("/v1/usage/history?days=100")

        assert response.status_code == 400


@pytest.mark.asyncio
class TestUsageBreakdown:
    """Tests for usage breakdown by endpoint."""

    async def test_get_usage_breakdown(self, client):
        """Test getting usage breakdown."""
        # Make some requests first
        await client.get("/v1/memories")
        await client.get("/v1/account")

        response = await client.get("/v1/usage/breakdown")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "endpoints" in data["data"]


@pytest.mark.asyncio
class TestUsageProjection:
    """Tests for usage projections."""

    async def test_get_projection(self, client):
        """Test getting usage projection."""
        response = await client.get("/v1/usage/projection")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "current" in data["data"]
        assert "projected" in data["data"]
        assert "days_elapsed" in data["data"]
        assert "days_remaining" in data["data"]


@pytest.mark.asyncio
class TestUsageExport:
    """Tests for usage data export."""

    async def test_export_csv(self, client):
        """Test CSV export."""
        response = await client.get("/v1/usage/export?format=csv")

        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers.get("content-disposition", "")

    async def test_export_json(self, client):
        """Test JSON export."""
        response = await client.get("/v1/usage/export?format=json")

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]


@pytest.mark.asyncio
class TestHealthCheck:
    """Tests for health check endpoint."""

    async def test_health_check(self, client):
        """Test health check endpoint."""
        # Remove auth for health check
        client.headers.pop("Authorization", None)

        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data["data"]
        assert "version" in data["data"]

    async def test_root_endpoint(self, client):
        """Test root endpoint."""
        client.headers.pop("Authorization", None)

        response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
