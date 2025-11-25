"""Storage layer for memory system.

This module provides a persistent storage solution for memories with metadata,
timestamps, embeddings, and efficient retrieval capabilities.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
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
        - embedding: JSON-encoded embedding vector (optional)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL,
                    embedding TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
            conn.commit()

            # Migration: add embedding column if it doesn't exist
            self._migrate_add_embedding_column(conn)

    def _migrate_add_embedding_column(self, conn: sqlite3.Connection) -> None:
        """Add embedding column to existing databases.

        This handles migration for databases created before embedding support.
        """
        try:
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]
            if "embedding" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
                conn.commit()
        except Exception:
            # Column might already exist or other issue, ignore
            pass

    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        """Add a new memory to storage.

        Args:
            text: The content of the memory to store.
            metadata: Optional dictionary of metadata to associate with the memory.
                Can contain any JSON-serializable data.
            memory_id: Optional custom ID for the memory. If not provided, a UUID
                will be generated automatically.
            embedding: Optional embedding vector for prefiltering. Should be a list
                of floats representing the semantic embedding of the text.

        Returns:
            The ID of the stored memory (either provided or generated).

        Raises:
            sqlite3.IntegrityError: If a memory with the given ID already exists.

        Example:
            >>> storage = MemoryStorage()
            >>> mid = storage.add_memory(
            ...     "The sky is blue",
            ...     metadata={"category": "fact", "confidence": 0.99},
            ...     embedding=[0.1, 0.2, 0.3, ...]
            ... )
            >>> isinstance(mid, str)
            True
        """
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        timestamp = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata or {})
        embedding_json = json.dumps(embedding) if embedding else None

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memories (id, text, metadata, timestamp, embedding) VALUES (?, ?, ?, ?, ?)",
                (memory_id, text, metadata_json, timestamp, embedding_json)
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
            - embedding: The embedding vector (or None if not set)

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
                "SELECT id, text, metadata, timestamp, embedding FROM memories WHERE id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            embedding = json.loads(row["embedding"]) if row["embedding"] else None
            return {
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "timestamp": row["timestamp"],
                "embedding": embedding
            }

    def get_all_memories(self, include_embeddings: bool = True) -> List[Dict[str, Any]]:
        """Retrieve all memories from storage.

        Args:
            include_embeddings: Whether to include embedding vectors. Default True.
                Set to False for faster retrieval when embeddings aren't needed.

        Returns:
            A list of memory dictionaries, each containing:
            - id: The memory ID
            - text: The memory content
            - metadata: The metadata dictionary
            - timestamp: ISO format timestamp string
            - embedding: The embedding vector (or None) if include_embeddings=True

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
            if include_embeddings:
                cursor = conn.execute(
                    "SELECT id, text, metadata, timestamp, embedding FROM memories ORDER BY timestamp DESC"
                )
                return [
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "metadata": json.loads(row["metadata"]),
                        "timestamp": row["timestamp"],
                        "embedding": json.loads(row["embedding"]) if row["embedding"] else None
                    }
                    for row in cursor.fetchall()
                ]
            else:
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
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """Update an existing memory.

        Args:
            memory_id: The ID of the memory to update.
            text: Optional new text content. If None, text is not updated.
            metadata: Optional new metadata. If None, metadata is not updated.
                Note: This replaces the entire metadata dictionary.
            embedding: Optional new embedding vector. If None, embedding is not updated.

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

        if embedding is not None:
            updates.append("embedding = ?")
            params.append(json.dumps(embedding))

        if not updates:
            return False

        params.append(memory_id)
        query = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0

    def update_embedding(self, memory_id: str, embedding: List[float]) -> bool:
        """Update just the embedding for a memory.

        This is a convenience method for updating embeddings without
        affecting other fields.

        Args:
            memory_id: The ID of the memory to update.
            embedding: The new embedding vector.

        Returns:
            True if updated successfully, False if memory not found.
        """
        return self.update_memory(memory_id, embedding=embedding)

    def get_memories_without_embeddings(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get memories that don't have embeddings yet.

        Useful for batch embedding generation.

        Args:
            limit: Maximum number of memories to return. None for all.
            offset: Number of memories to skip (for pagination).

        Returns:
            List of memory dictionaries without embeddings.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if limit is not None:
                cursor = conn.execute(
                    "SELECT id, text, metadata, timestamp FROM memories WHERE embedding IS NULL ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                )
            else:
                cursor = conn.execute(
                    "SELECT id, text, metadata, timestamp FROM memories WHERE embedding IS NULL ORDER BY timestamp DESC"
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

    def count_memories_without_embeddings(self) -> int:
        """Count memories that don't have embeddings yet.

        Returns:
            Number of memories without embeddings.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NULL")
            return cursor.fetchone()[0]

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

    def get_all_embeddings_as_array(self) -> Tuple[List[str], "np.ndarray"]:
        """Get all memory IDs and embeddings as a numpy array.

        This is more memory efficient than get_all_memories() when you only
        need embeddings for similarity computation.

        Returns:
            Tuple of (memory_ids, embeddings_array) where:
            - memory_ids: List of memory IDs (same order as embeddings)
            - embeddings_array: numpy array of shape (N, embedding_dim)
              Only includes memories that have embeddings.
        """
        import numpy as np

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL ORDER BY timestamp DESC"
            )
            rows = cursor.fetchall()

            if not rows:
                return [], np.array([])

            memory_ids = []
            embeddings = []
            for row in rows:
                memory_ids.append(row[0])
                embeddings.append(json.loads(row[1]))

            return memory_ids, np.array(embeddings, dtype=np.float32)

    def get_memories_by_ids(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple memories by their IDs.

        Args:
            memory_ids: List of memory IDs to retrieve.

        Returns:
            List of memory dictionaries (without embeddings to save memory).
            Order matches the input memory_ids list. Missing IDs are skipped.
        """
        if not memory_ids:
            return []

        placeholders = ','.join('?' * len(memory_ids))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"SELECT id, text, metadata, timestamp FROM memories WHERE id IN ({placeholders})",
                memory_ids
            )

            # Build a dict for fast lookup
            id_to_memory = {}
            for row in cursor.fetchall():
                id_to_memory[row["id"]] = {
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata"]),
                    "timestamp": row["timestamp"]
                }

            # Return in order of input IDs
            return [id_to_memory[mid] for mid in memory_ids if mid in id_to_memory]
