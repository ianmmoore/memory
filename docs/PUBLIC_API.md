# Memory System - Public API Guide

This document provides a customer-focused guide to integrating the Memory System into your applications.

## Overview

The Memory System provides intelligent context retrieval for LLM applications through two systems:

| System | Use Case | Key Features |
|--------|----------|--------------|
| **MemorySystem** | General purpose memory | Store facts, conversations, knowledge |
| **CodeMemorySystem** | Code intelligence | Index and query codebases |

Both systems use LLM-based scoring to find the most relevant memories for any query.

## Installation

```bash
pip install -r requirements.txt
```

```python
# Verify installation
from memory_lib import MemorySystem, CodeMemorySystem
print("Ready!")
```

## Quick Integration

### Step 1: Create LLM Functions

The system requires async functions that call your LLM provider:

```python
import openai

# Small model for scoring (fast/cheap)
async def small_model(prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

# Primary model for responses (powerful)
async def primary_model(prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content
```

### Step 2: Initialize Memory System

```python
from memory_lib import MemorySystem

memory = MemorySystem(
    small_model_fn=small_model,
    db_path="memories.db",        # SQLite database path
    relevance_threshold=0.7,      # Minimum relevance score (0-1)
    max_memories=10               # Maximum memories per query
)
```

### Step 3: Add Memories

```python
# Add simple memories
memory.add_memory("The user's name is Alice")
memory.add_memory("Alice prefers email communication")

# Add with metadata for filtering
memory.add_memory(
    "Alice's account was created on 2024-01-15",
    metadata={"category": "account", "user_id": "alice123"}
)
```

### Step 4: Query with Context

```python
# Simple query
response = await memory.query(
    context="User asking about their account",
    task="Help the user with their account question",
    primary_model_fn=primary_model
)
print(response)
```

## General Memory System API

### MemorySystem

```python
MemorySystem(
    small_model_fn: Callable[[str], Awaitable[str]],  # Required
    db_path: str = "memories.db",
    relevance_threshold: float = 0.7,
    max_memories: int = 10,
    batch_size: int = 10
)
```

### Core Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `add_memory(text, metadata=None)` | Store a memory | `str` (memory ID) |
| `get_memory(id)` | Retrieve by ID | `dict` or `None` |
| `get_all_memories()` | Get all memories | `list[dict]` |
| `update_memory(id, text=None, metadata=None)` | Update memory | `bool` |
| `delete_memory(id)` | Remove memory | `bool` |
| `retrieve_relevant_memories(context)` | Find relevant | `list[ScoredMemory]` |
| `query(context, task, primary_model_fn)` | Full query | `str` |

### Usage Examples

```python
# Add memory
mid = memory.add_memory("User prefers dark mode")

# Retrieve specific memory
mem = memory.get_memory(mid)
print(mem["text"])  # "User prefers dark mode"

# Find relevant memories
relevant = await memory.retrieve_relevant_memories(
    "What are the user's preferences?"
)
for r in relevant:
    print(f"{r.text} (score: {r.relevance_score:.2f})")

# Full query
response = await memory.query(
    context="User asking about UI settings",
    task="Explain their current settings",
    primary_model_fn=primary_model
)
```

## Code Memory System API

### CodeMemorySystem

```python
CodeMemorySystem(
    small_model_fn: Callable[[str], Awaitable[str]],  # Required
    db_path: str = "code_memories.db",
    relevance_threshold: float = 0.7,
    max_memories: int = 15,
    enable_caching: bool = True,
    enable_dependency_boost: bool = True,
    enable_recency_boost: bool = True
)
```

### Indexing Methods

| Method | Description |
|--------|-------------|
| `index_repository(dir, exclude_patterns=[])` | Index entire codebase |
| `index_file(path)` | Index single file |
| `reindex_file(path)` | Update file index |

### Query Methods

| Method | Description |
|--------|-------------|
| `retrieve_relevant_memories(context)` | Find relevant code |
| `query(context, primary_model_fn)` | Full code query |

### CodeContext

For code queries, use `CodeContext` to provide rich context:

```python
from memory_lib.codebase import CodeContext

context = CodeContext(
    user_query="How does authentication work?",  # Required
    current_file="src/auth/login.py",            # Optional
    errors="TypeError in validate_token",         # Optional
    recent_changes="Added OAuth support",         # Optional
    accessed_files=["auth.py", "utils.py"]       # Optional
)
```

### Usage Examples

```python
from memory_lib import CodeMemorySystem
from memory_lib.codebase import CodeContext

# Initialize
code_memory = CodeMemorySystem(
    small_model_fn=small_model,
    enable_caching=True
)

# Index codebase
code_memory.index_repository(
    "src/",
    exclude_patterns=["*/tests/*", "*/__pycache__/*"]
)

# Add documentation memory
code_memory.add_documentation_memory(
    title="Authentication Flow",
    content="Users authenticate via JWT tokens...",
    category="documentation"
)

# Query with context
context = CodeContext(
    user_query="Fix the login bug",
    current_file="src/auth/login.py",
    errors="Invalid token format"
)

response = await code_memory.query(context, primary_model_fn)
print(response)
```

## Configuration

### Recommended Settings

| Parameter | Use Case | Value |
|-----------|----------|-------|
| `relevance_threshold` | High precision | 0.8 |
| `relevance_threshold` | Balanced | 0.7 |
| `relevance_threshold` | High recall | 0.5 |
| `max_memories` | Chatbot | 5-10 |
| `max_memories` | Code assistant | 10-15 |
| `batch_size` | Default | 10 |
| `batch_size` | High throughput | 20 |

### Runtime Updates

```python
# Update configuration at runtime
memory.update_retrieval_config(
    relevance_threshold=0.75,
    max_memories=15
)
```

## Cost Estimation

| Memories | Small Model Calls | Estimated Cost |
|----------|-------------------|----------------|
| 10 | 10 | ~$0.008 |
| 100 | 100 | ~$0.08 |
| 500 | 500 | ~$0.40 |

With caching enabled (Code System), costs reduce by 60-80% after initial queries.

## Error Handling

```python
from memory_lib.exceptions import (
    MemoryNotFoundError,
    StorageError,
    RetrievalError
)

try:
    await memory.query(context, task, primary_model_fn)
except MemoryNotFoundError:
    print("Memory not found")
except RetrievalError as e:
    print(f"Retrieval failed: {e}")
except StorageError as e:
    print(f"Database error: {e}")
```

## Best Practices

### 1. Model Selection

- **Small model**: Use GPT-3.5-turbo, Claude Haiku, or similar
- **Primary model**: Use GPT-4, Claude Opus for best results

### 2. Memory Granularity

```python
# Good: Specific, atomic facts
memory.add_memory("User's timezone is PST")
memory.add_memory("User prefers weekly emails")

# Avoid: Long, combined facts
memory.add_memory("User is in PST timezone and prefers weekly emails on Monday mornings...")
```

### 3. Use Metadata

```python
# Add metadata for organization
memory.add_memory(
    "User upgraded to premium on 2024-01-15",
    metadata={
        "category": "subscription",
        "importance": "high",
        "user_id": "user123"
    }
)
```

### 4. Code System Optimization

```python
# Exclude non-essential files
code_memory.index_repository(
    "src/",
    exclude_patterns=[
        "*/tests/*",
        "*/__pycache__/*",
        "*/node_modules/*",
        "*/.git/*"
    ]
)
```

## Integration Examples

### Chatbot

```python
class MemoryBot:
    def __init__(self):
        self.memory = MemorySystem(small_model_fn=small_model)

    def learn(self, fact):
        self.memory.add_memory(fact)

    async def respond(self, message):
        return await self.memory.query(
            context=message,
            task=f"Respond helpfully to: {message}",
            primary_model_fn=primary_model
        )

# Usage
bot = MemoryBot()
bot.learn("User's favorite color is blue")
response = await bot.respond("What's my favorite color?")
```

### Code Assistant

```python
class CodeAssistant:
    def __init__(self, repo_path):
        self.memory = CodeMemorySystem(small_model_fn=small_model)
        self.memory.index_repository(repo_path)

    async def help(self, query, current_file=None, error=None):
        context = CodeContext(
            user_query=query,
            current_file=current_file,
            errors=error
        )
        return await self.memory.query(context, primary_model_fn)

# Usage
assistant = CodeAssistant("./src")
help_text = await assistant.help(
    "How do I add a new endpoint?",
    current_file="api/routes.py"
)
```

### Customer Support

```python
class SupportAgent:
    def __init__(self):
        self.memory = MemorySystem(small_model_fn=small_model)

    def load_knowledge(self, docs):
        for doc in docs:
            self.memory.add_memory(
                doc["content"],
                metadata={"source": doc["source"]}
            )

    async def answer(self, question):
        return await self.memory.query(
            context=question,
            task="Answer this customer question accurately",
            primary_model_fn=primary_model
        )
```

## Support

- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Architecture**: [01_ARCHITECTURE.md](01_ARCHITECTURE.md)

## Version

- **API Version**: 1.0
- **Python**: 3.9+
