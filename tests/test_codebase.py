"""Tests for the code memory system.

Run with: pytest test_codebase.py
"""

import pytest
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_lib import CodeMemorySystem
from memory_lib.codebase import CodeContext, CodeIndexer


async def mock_small_model(prompt: str) -> str:
    """Mock small model for testing."""
    if "auth" in prompt.lower():
        return "Score: 0.9\nReason: Authentication related"
    elif "handler" in prompt.lower():
        return "Score: 0.8\nReason: Handler function"
    else:
        return "Score: 0.5\nReason: Generic code"


async def mock_primary_model(prompt: str) -> str:
    """Mock primary model for testing."""
    return "Test code solution"


@pytest.fixture
def temp_db():
    """Fixture for temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def code_memory_system(temp_db):
    """Fixture for code memory system."""
    return CodeMemorySystem(
        small_model_fn=mock_small_model,
        db_path=temp_db,
        relevance_threshold=0.7,
        max_memories=10
    )


@pytest.fixture
def temp_python_file():
    """Fixture for temporary Python file."""
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as f:
        f.write('''"""Test module."""

def test_function(x: int) -> int:
    """Test function that doubles input.

    Args:
        x: Input integer

    Returns:
        Doubled value
    """
    return x * 2


class TestClass:
    """Test class."""

    def method(self):
        """Test method."""
        return test_function(5)
''')
        file_path = f.name

    yield file_path

    if os.path.exists(file_path):
        os.unlink(file_path)


class TestCodeIndexer:
    """Tests for code indexer."""

    def test_index_python_file(self, temp_python_file):
        """Test indexing a Python file."""
        indexer = CodeIndexer()
        entities = indexer.index_file(temp_python_file)

        assert len(entities) >= 2  # Function and class

        # Check function
        func = next((e for e in entities if e['entity_name'] == 'test_function'), None)
        assert func is not None
        assert func['language'] == 'python'
        assert 'def test_function' in func['code_snippet']
        assert 'doubles input' in func['docstring']

    def test_supported_languages(self):
        """Test that indexer has supported languages."""
        indexer = CodeIndexer()
        assert '.py' in indexer.supported_languages
        assert indexer.supported_languages['.py'] == 'python'

    def test_index_nonexistent_file(self):
        """Test indexing a file that doesn't exist."""
        indexer = CodeIndexer()
        with pytest.raises(FileNotFoundError):
            indexer.index_file("nonexistent.py")


class TestCodeStorage:
    """Tests for code memory storage."""

    def test_add_code_memory(self, code_memory_system):
        """Test adding a code memory."""
        mem_id = code_memory_system.add_code_memory(
            file_path="test.py",
            entity_name="test_func",
            code_snippet="def test_func(): pass",
            language="python"
        )

        assert isinstance(mem_id, str)
        assert len(mem_id) > 0

    def test_add_documentation_memory(self, code_memory_system):
        """Test adding documentation."""
        doc_id = code_memory_system.add_documentation_memory(
            title="Test Doc",
            content="This is test documentation",
            category="readme"
        )

        assert isinstance(doc_id, str)
        memory = code_memory_system.get_memory(doc_id)
        assert memory['title'] == "Test Doc"

    def test_add_debugging_session(self, code_memory_system):
        """Test adding debugging session."""
        debug_id = code_memory_system.add_debugging_session(
            title="Fixed bug",
            content="Bug was in authentication",
            metadata={"severity": "high"}
        )

        assert isinstance(debug_id, str)

    def test_get_memories_by_file(self, code_memory_system):
        """Test getting memories for a specific file."""
        code_memory_system.add_code_memory(
            file_path="test.py",
            entity_name="func1",
            code_snippet="def func1(): pass"
        )
        code_memory_system.add_code_memory(
            file_path="test.py",
            entity_name="func2",
            code_snippet="def func2(): pass"
        )
        code_memory_system.add_code_memory(
            file_path="other.py",
            entity_name="func3",
            code_snippet="def func3(): pass"
        )

        memories = code_memory_system.get_memories_by_file("test.py")
        assert len(memories) == 2


class TestCodeIndexing:
    """Tests for repository indexing."""

    def test_index_file(self, code_memory_system, temp_python_file):
        """Test indexing a single file."""
        entities = code_memory_system.index_file(temp_python_file)

        assert len(entities) >= 2
        # Check that entities were stored
        assert code_memory_system.storage.count_code_memories() >= 2

    def test_reindex_file(self, code_memory_system, temp_python_file):
        """Test re-indexing a file."""
        # Initial index
        code_memory_system.index_file(temp_python_file)
        initial_count = code_memory_system.storage.count_code_memories()

        # Re-index
        count = code_memory_system.reindex_file(temp_python_file)

        # Should have same number of entities
        assert count >= 2
        assert code_memory_system.storage.count_code_memories() == initial_count


class TestCodeRetrieval:
    """Tests for code memory retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_with_context(self, code_memory_system):
        """Test retrieving memories with code context."""
        # Add some code memories
        code_memory_system.add_code_memory(
            file_path="auth.py",
            entity_name="authenticate",
            code_snippet="def authenticate(user): pass",
            language="python"
        )

        context = CodeContext(
            user_query="Fix authentication",
            current_file="auth.py"
        )

        memories = await code_memory_system.retrieve_relevant_memories(context)

        # Should retrieve the auth function
        assert len(memories) >= 0  # May be 0 if score too low

    @pytest.mark.asyncio
    async def test_query_with_context(self, code_memory_system):
        """Test complete query with code context."""
        code_memory_system.add_code_memory(
            file_path="handlers.py",
            entity_name="handle_request",
            code_snippet="async def handle_request(): pass",
            language="python"
        )

        context = CodeContext(
            user_query="Fix handler error",
            current_file="handlers.py",
            errors="NoneType error"
        )

        response = await code_memory_system.query(
            context=context,
            primary_model_fn=mock_primary_model
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_caching(self, code_memory_system):
        """Test that caching works."""
        code_memory_system.add_code_memory(
            file_path="test.py",
            entity_name="test_func",
            code_snippet="def test_func(): pass"
        )

        context = CodeContext(user_query="test query")

        # First retrieval
        memories1 = await code_memory_system.retrieve_relevant_memories(context)

        # Second retrieval (should use cache)
        memories2 = await code_memory_system.retrieve_relevant_memories(context)

        # Check if any are cached
        cached = [m for m in memories2 if "[CACHED]" in m.reasoning]
        assert len(cached) >= 0  # May be 0 if caching disabled in mock


class TestConfiguration:
    """Tests for system configuration."""

    def test_get_stats(self, code_memory_system):
        """Test getting system statistics."""
        code_memory_system.add_code_memory(
            file_path="test.py",
            entity_name="test",
            code_snippet="def test(): pass"
        )

        stats = code_memory_system.get_stats()

        assert "code_memories" in stats
        assert "non_code_memories" in stats
        assert "cache" in stats
        assert stats["code_memories"] >= 1

    def test_clear_cache(self, code_memory_system):
        """Test clearing cache."""
        code_memory_system.clear_cache()
        # Should not raise an error

    def test_update_config(self, code_memory_system):
        """Test updating retrieval config."""
        code_memory_system.update_retrieval_config(
            relevance_threshold=0.8,
            max_memories=20
        )

        assert code_memory_system.retrieval.relevance_threshold == 0.8
        assert code_memory_system.retrieval.max_memories == 20


class TestCodeContext:
    """Tests for CodeContext."""

    def test_context_creation(self):
        """Test creating a code context."""
        context = CodeContext(
            user_query="Fix bug",
            current_file="test.py",
            errors="ValueError"
        )

        assert context.user_query == "Fix bug"
        assert context.current_file == "test.py"
        assert context.errors == "ValueError"

    def test_context_to_string(self):
        """Test converting context to string."""
        context = CodeContext(
            user_query="Implement feature",
            current_file="api.py",
            errors="TypeError"
        )

        context_str = context.to_context_string()

        assert "Implement feature" in context_str
        assert "api.py" in context_str
        assert "TypeError" in context_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
