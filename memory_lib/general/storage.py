"""Storage layer for memory system.

This module provides a persistent storage solution for memories with metadata,
timestamps, and efficient retrieval capabilities.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class MemoryStorage:
    """Persistent storage for memories using SQLite.

    This class provides a simple, efficient storage solution for memories with
    structured metadata. Each memory is stored with a unique ID, text content,
    metadata dictionary, and timestamp.

    Attributes:
        db_path: Path to the SQLite database file.

    Example:
        >>> storage = MemoryStorage("memories.db")
        >>> memory_id = storage.add_memory(
        ...     "Python uses dynamic typing",
        ...     metadata={"topic": "programming", "language": "python"}
        ... )
        >>> memories = storage.get_all_memories()
        >>> print(len(memories))
        1
    """

    def __init__(self, db_path: str = "memories.db"):
        """Initialize the memory storage.

        Args:
            db_path: Path to the SQLite database file. Will be created if it
                doesn't exist. Defaults to "memories.db".
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema.

        Creates the memories table if it doesn't exist with columns for:
        - id: Unique identifier (UUID)
        - text: The memory content
        - metadata: JSON-encoded metadata dictionary
        - timestamp: ISO format timestamp of creation
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
            conn.commit()

    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> str:
        """Add a new memory to storage.

        Args:
            text: The content of the memory to store.
            metadata: Optional dictionary of metadata to associate with the memory.
                Can contain any JSON-serializable data.
            memory_id: Optional custom ID for the memory. If not provided, a UUID
                will be generated automatically.

        Returns:
            The ID of the stored memory (either provided or generated).

        Raises:
            sqlite3.IntegrityError: If a memory with the given ID already exists.

        Example:
            >>> storage = MemoryStorage()
            >>> mid = storage.add_memory(
            ...     "The sky is blue",
            ...     metadata={"category": "fact", "confidence": 0.99}
            ... )
            >>> isinstance(mid, str)
            True
        """
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        timestamp = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata or {})

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memories (id, text, metadata, timestamp) VALUES (?, ?, ?, ?)",
                (memory_id, text, metadata_json, timestamp)
            )
            conn.commit()

        return memory_id

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: The unique identifier of the memory to retrieve.

        Returns:
            A dictionary containing the memory data with keys:
            - id: The memory ID
            - text: The memory content
            - metadata: The metadata dictionary
            - timestamp: ISO format timestamp string

            Returns None if the memory is not found.

        Example:
            >>> storage = MemoryStorage()
            >>> mid = storage.add_memory("Test memory")
            >>> memory = storage.get_memory(mid)
            >>> memory["text"]
            'Test memory'
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, text, metadata, timestamp FROM memories WHERE id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return {
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "timestamp": row["timestamp"]
            }

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Retrieve all memories from storage.

        Returns:
            A list of memory dictionaries, each containing:
            - id: The memory ID
            - text: The memory content
            - metadata: The metadata dictionary
            - timestamp: ISO format timestamp string

            Memories are ordered by timestamp (most recent first).

        Example:
            >>> storage = MemoryStorage()
            >>> storage.add_memory("First memory")
            >>> storage.add_memory("Second memory")
            >>> len(storage.get_all_memories())
            2
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, text, metadata, timestamp FROM memories ORDER BY timestamp DESC"
            )

            return [
                {
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata"]),
                    "timestamp": row["timestamp"]
                }
                for row in cursor.fetchall()
            ]

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory.

        Args:
            memory_id: The ID of the memory to update.
            text: Optional new text content. If None, text is not updated.
            metadata: Optional new metadata. If None, metadata is not updated.
                Note: This replaces the entire metadata dictionary.

        Returns:
            True if the memory was found and updated, False otherwise.

        Example:
            >>> storage = MemoryStorage()
            >>> mid = storage.add_memory("Original text")
            >>> storage.update_memory(mid, text="Updated text")
            True
            >>> storage.get_memory(mid)["text"]
            'Updated text'
        """
        updates = []
        params = []

        if text is not None:
            updates.append("text = ?")
            params.append(text)

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if not updates:
            return False

        params.append(memory_id)
        query = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from storage.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if the memory was found and deleted, False otherwise.

        Example:
            >>> storage = MemoryStorage()
            >>> mid = storage.add_memory("Temporary memory")
            >>> storage.delete_memory(mid)
            True
            >>> storage.get_memory(mid) is None
            True
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    def count_memories(self) -> int:
        """Get the total number of memories in storage.

        Returns:
            The count of all memories.

        Example:
            >>> storage = MemoryStorage()
            >>> storage.add_memory("Memory 1")
            >>> storage.add_memory("Memory 2")
            >>> storage.count_memories()
            2
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            return cursor.fetchone()[0]

    def clear_all(self) -> int:
        """Delete all memories from storage.

        Returns:
            The number of memories deleted.

        Warning:
            This operation is irreversible. All memories will be permanently deleted.

        Example:
            >>> storage = MemoryStorage()
            >>> storage.add_memory("Memory 1")
            >>> storage.add_memory("Memory 2")
            >>> storage.clear_all()
            2
            >>> storage.count_memories()
            0
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memories")
            conn.commit()
            return cursor.rowcount
