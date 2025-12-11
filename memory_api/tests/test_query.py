"""
Tests for query and retrieval endpoints.

Tests:
- Memory retrieval
- Query with answer generation
- Memory extraction
- Relevance scoring
"""

import pytest


@pytest.mark.asyncio
class TestQueryMemories:
    """Tests for memory query endpoint."""

    async def test_query_empty_memories(self, client, sample_query):
        """Test query with no memories."""
        response = await client.post("/v1/query", json=sample_query)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["memories"] == []
        assert data["data"]["total_memories_searched"] == 0

    async def test_query_with_memories(self, client, sample_memories, sample_query):
        """Test query with existing memories."""
        # Create memories
        for memory in sample_memories:
            await client.post("/v1/memories", json=memory)

        # Query
        response = await client.post("/v1/query", json=sample_query)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total_memories_searched"] == len(sample_memories)

    async def test_query_relevance_threshold(self, client, sample_memories):
        """Test that relevance threshold filters results."""
        # Create memories
        for memory in sample_memories:
            await client.post("/v1/memories", json=memory)

        # Query with high threshold
        response = await client.post(
            "/v1/query",
            json={
                "context": "Python programming",
                "relevance_threshold": 0.9,
            }
        )

        assert response.status_code == 200
        # High threshold should return fewer or no results
        data = response.json()
        assert data["success"] is True

    async def test_query_max_memories(self, client, sample_memories):
        """Test max_memories parameter."""
        # Create memories
        for memory in sample_memories:
            await client.post("/v1/memories", json=memory)

        # Query with limit
        response = await client.post(
            "/v1/query",
            json={
                "context": "user information",
                "max_memories": 1,
                "relevance_threshold": 0.0,  # Low threshold to get results
            }
        )

        assert response.status_code == 200
        data = response.json()
        # Should return at most 1 memory
        assert len(data["data"]["memories"]) <= 1

    async def test_query_includes_metadata(self, client, sample_memory):
        """Test that query includes metadata when requested."""
        # Create memory with metadata
        await client.post("/v1/memories", json=sample_memory)

        # Query with metadata
        response = await client.post(
            "/v1/query",
            json={
                "context": "dark mode",
                "include_metadata": True,
                "relevance_threshold": 0.0,
            }
        )

        assert response.status_code == 200
        # If there are results, they should include metadata
        memories = response.json()["data"]["memories"]
        for memory in memories:
            assert "metadata" in memory

    async def test_query_excludes_metadata(self, client, sample_memory):
        """Test that query excludes metadata when requested."""
        # Create memory
        await client.post("/v1/memories", json=sample_memory)

        # Query without metadata
        response = await client.post(
            "/v1/query",
            json={
                "context": "dark mode",
                "include_metadata": False,
                "relevance_threshold": 0.0,
            }
        )

        assert response.status_code == 200
        # Results should have None metadata
        memories = response.json()["data"]["memories"]
        for memory in memories:
            assert memory["metadata"] is None


@pytest.mark.asyncio
class TestQueryWithAnswer:
    """Tests for query with answer generation."""

    async def test_query_answer_no_memories(self, client):
        """Test answer generation with no memories."""
        response = await client.post(
            "/v1/query/answer",
            json={"context": "What is the user's name?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "answer" in data["data"]
        assert data["data"]["memories_used"] == []

    async def test_query_answer_with_memories(self, client, sample_memories):
        """Test answer generation with memories."""
        # Create memories
        for memory in sample_memories:
            await client.post("/v1/memories", json=memory)

        response = await client.post(
            "/v1/query/answer",
            json={
                "context": "What does the user do for work?",
                "relevance_threshold": 0.0,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "answer" in data["data"]
        assert "model" in data["data"]

    async def test_query_answer_custom_prompt(self, client, sample_memory):
        """Test answer with custom prompt template."""
        await client.post("/v1/memories", json=sample_memory)

        response = await client.post(
            "/v1/query/answer",
            json={
                "context": "UI preferences",
                "prompt_template": "Based on {memories}, answer: {context}",
                "relevance_threshold": 0.0,
            }
        )

        assert response.status_code == 200


@pytest.mark.asyncio
class TestMemoryExtraction:
    """Tests for memory extraction endpoint."""

    async def test_extract_from_conversation(self, client):
        """Test extracting memories from conversation."""
        conversation = """
        User: Hi, I'm John and I work as a data scientist.
        Assistant: Nice to meet you, John! What programming languages do you use?
        User: Mostly Python and R. I prefer Python for machine learning.
        """

        response = await client.post(
            "/v1/query/extract",
            json={
                "content": conversation,
                "content_type": "conversation",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "extracted_count" in data["data"]
        assert "memory_ids" in data["data"]

    async def test_extract_with_metadata(self, client):
        """Test extraction with custom metadata."""
        response = await client.post(
            "/v1/query/extract",
            json={
                "content": "The user mentioned they live in New York.",
                "content_type": "notes",
                "metadata": {"source": "meeting", "date": "2024-01-15"},
            }
        )

        assert response.status_code == 200

        # Check that created memories have the metadata
        if response.json()["data"]["extracted_count"] > 0:
            memory_id = response.json()["data"]["memory_ids"][0]
            memory_response = await client.get(f"/v1/memories/{memory_id}")
            metadata = memory_response.json()["data"]["metadata"]
            assert metadata.get("source") == "meeting"


@pytest.mark.asyncio
class TestQueryValidation:
    """Tests for query input validation."""

    async def test_query_empty_context(self, client):
        """Test that empty context is rejected."""
        response = await client.post(
            "/v1/query",
            json={"context": ""}
        )

        assert response.status_code == 400

    async def test_query_invalid_threshold(self, client):
        """Test that invalid threshold is rejected."""
        response = await client.post(
            "/v1/query",
            json={"context": "test", "relevance_threshold": 1.5}
        )

        assert response.status_code == 400

    async def test_query_invalid_max_memories(self, client):
        """Test that invalid max_memories is rejected."""
        response = await client.post(
            "/v1/query",
            json={"context": "test", "max_memories": 0}
        )

        assert response.status_code == 400

        response = await client.post(
            "/v1/query",
            json={"context": "test", "max_memories": 200}
        )

        assert response.status_code == 400
