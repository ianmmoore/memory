"""Storage layer for code memories with specialized schema.

This module extends the general storage to support code-specific metadata including
file paths, entity names, code snippets, signatures, and dependency information.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class CodeMemoryStorage:
    """Persistent storage for code memories with specialized schema.

    This class extends the general memory storage concept with code-specific
    fields and indices. It stores memories about code entities (functions, classes,
    configuration files, documentation, debugging sessions).

    Attributes:
        db_path: Path to the SQLite database file.

    Example:
        >>> storage = CodeMemoryStorage("code_memories.db")
        >>> memory_id = storage.add_code_memory(
        ...     file_path="src/utils/parser.py",
        ...     entity_name="parse_json",
        ...     code_snippet="def parse_json(data: str) -> dict:\\n    return json.loads(data)",
        ...     docstring="Parse JSON string into dictionary",
        ...     signature="parse_json(data: str) -> dict",
        ...     metadata={"language": "python", "complexity": "low"}
        ... )
    """

    def __init__(self, db_path: str = "code_memories.db"):
        """Initialize the code memory storage.

        Args:
            db_path: Path to the SQLite database file. Will be created if it
                doesn't exist. Defaults to "code_memories.db".
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema for code memories.

        Creates tables for:
        - code_memories: Main table for code entities
        - non_code_memories: Table for documentation, debugging sessions, etc.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Code memories table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS code_memories (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    entity_name TEXT,
                    code_snippet TEXT,
                    docstring TEXT,
                    signature TEXT,
                    language TEXT,
                    dependencies TEXT,
                    imports TEXT,
                    complexity TEXT,
                    last_modified TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            # Non-code memories table (docs, debugging sessions, etc.)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS non_code_memories (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    file_path TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            # Indices for efficient retrieval
            conn.execute("CREATE INDEX IF NOT EXISTS idx_code_file_path ON code_memories(file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_code_entity ON code_memories(entity_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_code_language ON code_memories(language)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_code_modified ON code_memories(last_modified)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_noncode_category ON non_code_memories(category)")

            conn.commit()

    def add_code_memory(
        self,
        file_path: str,
        entity_name: Optional[str] = None,
        code_snippet: Optional[str] = None,
        docstring: Optional[str] = None,
        signature: Optional[str] = None,
        language: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        imports: Optional[List[str]] = None,
        complexity: Optional[str] = None,
        last_modified: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> str:
        """Add a code memory to storage.

        Args:
            file_path: Path to the source file (relative or absolute).
            entity_name: Name of the function, class, or code block.
            code_snippet: The actual code content.
            docstring: Documentation string for the entity.
            signature: Function/method signature.
            language: Programming language (e.g., "python", "javascript").
            dependencies: List of other entities this code depends on.
            imports: List of import statements used.
            complexity: Complexity indicator ("low", "medium", "high").
            last_modified: ISO timestamp of last file modification.
            metadata: Additional metadata dictionary.
            memory_id: Optional custom ID. If not provided, UUID is generated.

        Returns:
            The ID of the stored memory.

        Example:
            >>> storage = CodeMemoryStorage()
            >>> mid = storage.add_code_memory(
            ...     file_path="api/handlers.py",
            ...     entity_name="handle_request",
            ...     code_snippet="async def handle_request(req): ...",
            ...     language="python",
            ...     dependencies=["validate_input", "process_data"],
            ...     complexity="medium"
            ... )
        """
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        timestamp = datetime.utcnow().isoformat()
        dependencies_json = json.dumps(dependencies or [])
        imports_json = json.dumps(imports or [])
        metadata_json = json.dumps(metadata or {})

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO code_memories (
                    id, file_path, entity_name, code_snippet, docstring,
                    signature, language, dependencies, imports, complexity,
                    last_modified, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id, file_path, entity_name, code_snippet, docstring,
                signature, language, dependencies_json, imports_json,
                complexity, last_modified or timestamp, timestamp, metadata_json
            ))
            conn.commit()

        return memory_id

    def add_non_code_memory(
        self,
        category: str,
        content: str,
        title: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> str:
        """Add a non-code memory (documentation, debugging session, etc.).

        Args:
            category: Type of memory (e.g., "readme", "architecture", "debugging",
                "design-decision", "api-docs").
            content: The actual content of the memory.
            title: Optional title or summary.
            file_path: Optional associated file path.
            metadata: Additional metadata dictionary.
            memory_id: Optional custom ID.

        Returns:
            The ID of the stored memory.

        Example:
            >>> storage = CodeMemoryStorage()
            >>> mid = storage.add_non_code_memory(
            ...     category="debugging",
            ...     title="Fixed async race condition",
            ...     content="Issue: Multiple async tasks...",
            ...     metadata={"severity": "high", "date": "2024-01-15"}
            ... )
        """
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        timestamp = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata or {})

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO non_code_memories (
                    id, category, title, content, file_path, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (memory_id, category, title, content, file_path, timestamp, metadata_json))
            conn.commit()

        return memory_id

    def get_code_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific code memory by ID.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            Dictionary with all code memory fields, or None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM code_memories WHERE id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_code_dict(row)

    def get_non_code_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific non-code memory by ID.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            Dictionary with all non-code memory fields, or None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM non_code_memories WHERE id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_noncode_dict(row)

    def get_all_code_memories(self) -> List[Dict[str, Any]]:
        """Retrieve all code memories.

        Returns:
            List of code memory dictionaries, ordered by timestamp (most recent first).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM code_memories ORDER BY timestamp DESC"
            )
            return [self._row_to_code_dict(row) for row in cursor.fetchall()]

    def get_all_non_code_memories(self) -> List[Dict[str, Any]]:
        """Retrieve all non-code memories.

        Returns:
            List of non-code memory dictionaries, ordered by timestamp (most recent first).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM non_code_memories ORDER BY timestamp DESC"
            )
            return [self._row_to_noncode_dict(row) for row in cursor.fetchall()]

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Retrieve all memories (both code and non-code).

        Returns:
            List of all memory dictionaries with a 'type' field indicating 'code' or 'non-code'.
        """
        code_memories = [
            {**mem, "type": "code"} for mem in self.get_all_code_memories()
        ]
        non_code_memories = [
            {**mem, "type": "non-code"} for mem in self.get_all_non_code_memories()
        ]

        # Combine and sort by timestamp
        all_memories = code_memories + non_code_memories
        all_memories.sort(key=lambda x: x["timestamp"], reverse=True)

        return all_memories

    def get_memories_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all code memories for a specific file.

        Args:
            file_path: Path to the source file.

        Returns:
            List of code memory dictionaries for the specified file.

        Example:
            >>> memories = storage.get_memories_by_file("src/api/handlers.py")
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM code_memories WHERE file_path = ? ORDER BY timestamp DESC",
                (file_path,)
            )
            return [self._row_to_code_dict(row) for row in cursor.fetchall()]

    def get_memories_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Get all code memories for a specific language.

        Args:
            language: Programming language (e.g., "python", "javascript").

        Returns:
            List of code memory dictionaries for the specified language.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM code_memories WHERE language = ? ORDER BY timestamp DESC",
                (language,)
            )
            return [self._row_to_code_dict(row) for row in cursor.fetchall()]

    def get_memories_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all non-code memories for a specific category.

        Args:
            category: Category name (e.g., "debugging", "architecture").

        Returns:
            List of non-code memory dictionaries for the specified category.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM non_code_memories WHERE category = ? ORDER BY timestamp DESC",
                (category,)
            )
            return [self._row_to_noncode_dict(row) for row in cursor.fetchall()]

    def update_code_memory(
        self,
        memory_id: str,
        **kwargs
    ) -> bool:
        """Update an existing code memory.

        Args:
            memory_id: The ID of the memory to update.
            **kwargs: Fields to update (e.g., code_snippet="...", language="python").

        Returns:
            True if updated successfully, False if memory not found.

        Example:
            >>> storage.update_code_memory(
            ...     mid,
            ...     code_snippet="def new_code(): pass",
            ...     complexity="low"
            ... )
        """
        if not kwargs:
            return False

        # Handle list fields that need JSON encoding
        list_fields = {"dependencies", "imports"}
        for field in list_fields:
            if field in kwargs and isinstance(kwargs[field], list):
                kwargs[field] = json.dumps(kwargs[field])

        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = json.dumps(kwargs["metadata"])

        # Build update query
        updates = [f"{key} = ?" for key in kwargs.keys()]
        values = list(kwargs.values())
        values.append(memory_id)

        query = f"UPDATE code_memories SET {', '.join(updates)} WHERE id = ?"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory (code or non-code) by ID.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if deleted successfully, False if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM code_memories WHERE id = ?", (memory_id,))
            if cursor.rowcount > 0:
                conn.commit()
                return True

            cursor = conn.execute("DELETE FROM non_code_memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    def count_code_memories(self) -> int:
        """Get the total number of code memories."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM code_memories")
            return cursor.fetchone()[0]

    def count_non_code_memories(self) -> int:
        """Get the total number of non-code memories."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM non_code_memories")
            return cursor.fetchone()[0]

    def count_memories(self) -> int:
        """Get the total number of all memories."""
        return self.count_code_memories() + self.count_non_code_memories()

    def clear_all(self) -> int:
        """Delete all memories (both code and non-code).

        Returns:
            The total number of memories deleted.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM code_memories")
            code_count = cursor.rowcount
            cursor = conn.execute("DELETE FROM non_code_memories")
            noncode_count = cursor.rowcount
            conn.commit()
            return code_count + noncode_count

    def _row_to_code_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a code memory row to a dictionary."""
        return {
            "id": row["id"],
            "file_path": row["file_path"],
            "entity_name": row["entity_name"],
            "code_snippet": row["code_snippet"],
            "docstring": row["docstring"],
            "signature": row["signature"],
            "language": row["language"],
            "dependencies": json.loads(row["dependencies"]) if row["dependencies"] else [],
            "imports": json.loads(row["imports"]) if row["imports"] else [],
            "complexity": row["complexity"],
            "last_modified": row["last_modified"],
            "timestamp": row["timestamp"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
        }

    def _row_to_noncode_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a non-code memory row to a dictionary."""
        return {
            "id": row["id"],
            "category": row["category"],
            "title": row["title"],
            "content": row["content"],
            "file_path": row["file_path"],
            "timestamp": row["timestamp"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
        }
