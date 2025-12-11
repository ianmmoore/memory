"""
Tests for memory CRUD operations.

Tests:
- Create memory
- Read memory (single and list)
- Update memory
- Delete memory
- Pagination
- Error handling
"""

import pytest


@pytest.mark.asyncio
class TestCreateMemory:
    """Tests for memory creation."""

    async def test_create_memory_success(self, client, sample_memory):
        """Test successful memory creation."""
        response = await client.post("/v1/memories", json=sample_memory)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["text"] == sample_memory["text"]
        assert data["data"]["metadata"] == sample_memory["metadata"]
        assert "id" in data["data"]
        assert "created_at" in data["data"]

    async def test_create_memory_minimal(self, client):
        """Test memory creation with minimal data."""
        response = await client.post(
            "/v1/memories",
            json={"text": "Simple memory"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["data"]["text"] == "Simple memory"
        assert data["data"]["metadata"] == {}

    async def test_create_memory_custom_id(self, client):
        """Test memory creation with custom ID."""
        custom_id = "my_custom_memory_id"
        response = await client.post(
            "/v1/memories",
            json={"text": "Memory with custom ID", "memory_id": custom_id}
        )

        assert response.status_code == 201
        assert response.json()["data"]["id"] == custom_id

    async def test_create_memory_duplicate_id(self, client):
        """Test that duplicate custom IDs are rejected."""
        custom_id = "duplicate_id_test"

        # Create first memory
        await client.post(
            "/v1/memories",
            json={"text": "First memory", "memory_id": custom_id}
        )

        # Try to create second with same ID
        response = await client.post(
            "/v1/memories",
            json={"text": "Second memory", "memory_id": custom_id}
        )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "MEMORY_ID_EXISTS"

    async def test_create_memory_empty_text(self, client):
        """Test that empty text is rejected."""
        response = await client.post(
            "/v1/memories",
            json={"text": ""}
        )

        assert response.status_code == 400
        assert "VALIDATION" in response.json()["error"]["code"]

    async def test_create_memory_whitespace_only(self, client):
        """Test that whitespace-only text is rejected."""
        response = await client.post(
            "/v1/memories",
            json={"text": "   \n\t  "}
        )

        assert response.status_code == 400


@pytest.mark.asyncio
class TestReadMemory:
    """Tests for reading memories."""

    async def test_get_memory_success(self, client, sample_memory):
        """Test successful memory retrieval."""
        # Create memory
        create_response = await client.post("/v1/memories", json=sample_memory)
        memory_id = create_response.json()["data"]["id"]

        # Get memory
        response = await client.get(f"/v1/memories/{memory_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == memory_id
        assert data["data"]["text"] == sample_memory["text"]

    async def test_get_memory_not_found(self, client):
        """Test 404 for non-existent memory."""
        response = await client.get("/v1/memories/nonexistent_id")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "MEMORY_NOT_FOUND"

    async def test_list_memories_empty(self, client):
        """Test listing memories when empty."""
        response = await client.get("/v1/memories")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []
        assert data["pagination"]["total"] == 0

    async def test_list_memories_with_data(self, client, sample_memories):
        """Test listing memories with data."""
        # Create memories
        for memory in sample_memories:
            await client.post("/v1/memories", json=memory)

        # List memories
        response = await client.get("/v1/memories")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == len(sample_memories)
        assert data["pagination"]["total"] == len(sample_memories)

    async def test_list_memories_pagination(self, client, sample_memories):
        """Test memory pagination."""
        # Create memories
        for memory in sample_memories:
            await client.post("/v1/memories", json=memory)

        # Get first page
        response = await client.get("/v1/memories?page=1&per_page=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["per_page"] == 2
        assert data["pagination"]["has_next"] is True

    async def test_count_memories(self, client, sample_memories):
        """Test memory count endpoint."""
        # Create memories
        for memory in sample_memories:
            await client.post("/v1/memories", json=memory)

        response = await client.get("/v1/memories/count")

        assert response.status_code == 200
        assert response.json()["data"]["count"] == len(sample_memories)


@pytest.mark.asyncio
class TestUpdateMemory:
    """Tests for updating memories."""

    async def test_update_memory_text(self, client, sample_memory):
        """Test updating memory text."""
        # Create memory
        create_response = await client.post("/v1/memories", json=sample_memory)
        memory_id = create_response.json()["data"]["id"]

        # Update memory
        response = await client.put(
            f"/v1/memories/{memory_id}",
            json={"text": "Updated text"}
        )

        assert response.status_code == 200
        assert response.json()["data"]["text"] == "Updated text"

    async def test_update_memory_metadata(self, client, sample_memory):
        """Test updating memory metadata."""
        # Create memory
        create_response = await client.post("/v1/memories", json=sample_memory)
        memory_id = create_response.json()["data"]["id"]

        # Update metadata
        new_metadata = {"category": "updated", "new_field": "value"}
        response = await client.put(
            f"/v1/memories/{memory_id}",
            json={"metadata": new_metadata}
        )

        assert response.status_code == 200
        assert response.json()["data"]["metadata"] == new_metadata

    async def test_update_memory_not_found(self, client):
        """Test updating non-existent memory."""
        response = await client.put(
            "/v1/memories/nonexistent_id",
            json={"text": "Updated"}
        )

        assert response.status_code == 404


@pytest.mark.asyncio
class TestDeleteMemory:
    """Tests for deleting memories."""

    async def test_delete_memory_success(self, client, sample_memory):
        """Test successful memory deletion."""
        # Create memory
        create_response = await client.post("/v1/memories", json=sample_memory)
        memory_id = create_response.json()["data"]["id"]

        # Delete memory
        response = await client.delete(f"/v1/memories/{memory_id}")

        assert response.status_code == 200
        assert response.json()["data"]["deleted"] is True

        # Verify deleted
        get_response = await client.get(f"/v1/memories/{memory_id}")
        assert get_response.status_code == 404

    async def test_delete_memory_not_found(self, client):
        """Test deleting non-existent memory."""
        response = await client.delete("/v1/memories/nonexistent_id")

        assert response.status_code == 404

    async def test_delete_all_memories_requires_confirm(self, client, sample_memories):
        """Test that delete all requires confirmation."""
        # Create memories
        for memory in sample_memories:
            await client.post("/v1/memories", json=memory)

        # Try without confirm
        response = await client.delete("/v1/memories?confirm=false")

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "CONFIRMATION_REQUIRED"

    async def test_delete_all_memories_success(self, client, sample_memories):
        """Test successful delete all."""
        # Create memories
        for memory in sample_memories:
            await client.post("/v1/memories", json=memory)

        # Delete all
        response = await client.delete("/v1/memories?confirm=true")

        assert response.status_code == 200
        assert response.json()["data"]["deleted"] is True
        assert response.json()["data"]["count"] == len(sample_memories)


@pytest.mark.asyncio
class TestMemoryMetadata:
    """Tests for memory metadata handling."""

    async def test_complex_metadata(self, client):
        """Test memory with complex nested metadata."""
        complex_metadata = {
            "category": "test",
            "tags": ["tag1", "tag2"],
            "nested": {
                "key": "value",
                "number": 42,
            },
            "boolean": True,
        }

        response = await client.post(
            "/v1/memories",
            json={"text": "Test", "metadata": complex_metadata}
        )

        assert response.status_code == 201
        assert response.json()["data"]["metadata"] == complex_metadata

    async def test_metadata_null_values(self, client):
        """Test metadata with null values."""
        response = await client.post(
            "/v1/memories",
            json={"text": "Test", "metadata": {"nullable": None}}
        )

        assert response.status_code == 201
        assert response.json()["data"]["metadata"]["nullable"] is None
