"""Tests for the general memory system.

Run with: pytest test_general.py
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_lib import MemorySystem


# Mock LLM for testing
async def mock_small_model(prompt: str) -> str:
    """Mock small model that returns predictable scores."""
    if "test" in prompt.lower():
        return "Score: 0.9\nReason: Contains test keyword"
    elif "example" in prompt.lower():
        return "Score: 0.75\nReason: Contains example keyword"
    else:
        return "Score: 0.4\nReason: Generic content"


async def mock_primary_model(prompt: str) -> str:
    """Mock primary model."""
    return "Test response from primary model"


@pytest.fixture
def temp_db():
    """Fixture to create a temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def memory_system(temp_db):
    """Fixture to create a memory system instance."""
    return MemorySystem(
        small_model_fn=mock_small_model,
        db_path=temp_db,
        relevance_threshold=0.7,
        max_memories=5
    )


class TestMemoryStorage:
    """Test cases for memory storage."""

    def test_add_memory(self, memory_system):
        """Test adding a memory."""
        mem_id = memory_system.add_memory(
            "This is a test memory",
            metadata={"category": "test"}
        )
        assert isinstance(mem_id, str)
        assert len(mem_id) > 0

    def test_get_memory(self, memory_system):
        """Test retrieving a memory."""
        mem_id = memory_system.add_memory("Test memory")
        memory = memory_system.get_memory(mem_id)

        assert memory is not None
        assert memory["id"] == mem_id
        assert memory["text"] == "Test memory"

    def test_get_all_memories(self, memory_system):
        """Test retrieving all memories."""
        memory_system.add_memory("Memory 1")
        memory_system.add_memory("Memory 2")
        memory_system.add_memory("Memory 3")

        all_memories = memory_system.get_all_memories()
        assert len(all_memories) == 3

    def test_update_memory(self, memory_system):
        """Test updating a memory."""
        mem_id = memory_system.add_memory("Original text")

        success = memory_system.update_memory(mem_id, text="Updated text")
        assert success is True

        memory = memory_system.get_memory(mem_id)
        assert memory["text"] == "Updated text"

    def test_delete_memory(self, memory_system):
        """Test deleting a memory."""
        mem_id = memory_system.add_memory("To be deleted")

        success = memory_system.delete_memory(mem_id)
        assert success is True

        memory = memory_system.get_memory(mem_id)
        assert memory is None

    def test_count_memories(self, memory_system):
        """Test counting memories."""
        assert memory_system.count_memories() == 0

        memory_system.add_memory("Memory 1")
        memory_system.add_memory("Memory 2")

        assert memory_system.count_memories() == 2


class TestMemoryRetrieval:
    """Test cases for memory retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories(self, memory_system):
        """Test retrieving relevant memories."""
        # Add test memories
        memory_system.add_memory("This is a test memory", metadata={"type": "test"})
        memory_system.add_memory("This is an example", metadata={"type": "example"})
        memory_system.add_memory("Generic content", metadata={"type": "generic"})

        # Retrieve with test context
        memories = await memory_system.retrieve_relevant_memories("test context")

        # Should return memories with score >= 0.7
        assert len(memories) >= 1
        assert all(m.relevance_score >= 0.7 for m in memories)

    @pytest.mark.asyncio
    async def test_format_memories(self, memory_system):
        """Test formatting memories for prompt."""
        memory_system.add_memory("Test memory")

        memories = await memory_system.retrieve_relevant_memories("test")
        formatted = memory_system.format_memories_for_prompt(memories)

        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "<memory" in formatted

    @pytest.mark.asyncio
    async def test_query(self, memory_system):
        """Test complete query pipeline."""
        memory_system.add_memory("Test memory about Python")

        response = await memory_system.query(
            context="Tell me about Python",
            task="Explain Python features",
            primary_model_fn=mock_primary_model
        )

        assert isinstance(response, str)
        assert len(response) > 0


class TestConfiguration:
    """Test cases for system configuration."""

    def test_update_retrieval_config(self, memory_system):
        """Test updating retrieval configuration."""
        memory_system.update_retrieval_config(
            relevance_threshold=0.8,
            max_memories=10
        )

        assert memory_system.retrieval.relevance_threshold == 0.8
        assert memory_system.retrieval.max_memories == 10

    def test_get_stats(self, memory_system):
        """Test getting system statistics."""
        memory_system.add_memory("Test")

        stats = memory_system.get_stats()

        assert "total_memories" in stats
        assert "retrieval_config" in stats
        assert stats["total_memories"] == 1

    def test_clear_all_memories(self, memory_system):
        """Test clearing all memories."""
        memory_system.add_memory("Memory 1")
        memory_system.add_memory("Memory 2")

        count = memory_system.clear_all_memories()

        assert count == 2
        assert memory_system.count_memories() == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_get_nonexistent_memory(self, memory_system):
        """Test getting a memory that doesn't exist."""
        memory = memory_system.get_memory("nonexistent-id")
        assert memory is None

    def test_delete_nonexistent_memory(self, memory_system):
        """Test deleting a memory that doesn't exist."""
        success = memory_system.delete_memory("nonexistent-id")
        assert success is False

    def test_update_nonexistent_memory(self, memory_system):
        """Test updating a memory that doesn't exist."""
        success = memory_system.update_memory(
            "nonexistent-id",
            text="New text"
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_retrieve_with_no_memories(self, memory_system):
        """Test retrieval when no memories exist."""
        memories = await memory_system.retrieve_relevant_memories("test")
        assert len(memories) == 0

    def test_invalid_threshold(self, temp_db):
        """Test initialization with invalid threshold."""
        with pytest.raises(ValueError):
            MemorySystem(
                small_model_fn=mock_small_model,
                db_path=temp_db,
                relevance_threshold=1.5  # Invalid
            )

    def test_invalid_max_memories(self, temp_db):
        """Test initialization with invalid max_memories."""
        with pytest.raises(ValueError):
            MemorySystem(
                small_model_fn=mock_small_model,
                db_path=temp_db,
                max_memories=0  # Invalid
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
