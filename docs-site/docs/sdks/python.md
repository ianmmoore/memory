---
sidebar_position: 1
---

# Python SDK

Official Python SDK for Memory API.

## Installation

```bash
pip install memory-api
```

## Quick Start

```python
from memory_api import MemoryClient

# Initialize client
client = MemoryClient(api_key="mem_live_your_key")

# Create a memory
memory = client.memories.create(
    text="User prefers dark mode interfaces",
    metadata={"category": "preference"}
)
print(f"Created: {memory.id}")

# Query memories
results = client.query(
    context="What are the user's UI preferences?"
)

for mem in results.memories:
    print(f"[{mem.relevance_score:.2f}] {mem.text}")

# Get an answer
answer = client.query_answer(
    context="What theme should I use for this user?"
)
print(answer.answer)
```

## Client Configuration

```python
from memory_api import MemoryClient

client = MemoryClient(
    api_key="mem_live_...",
    base_url="https://api.memory-api.com",  # Optional
    timeout=30,  # Request timeout in seconds
)
```

## Memory Operations

### Create

```python
memory = client.memories.create(
    text="Memory content",
    metadata={"key": "value"},
    memory_id="optional_custom_id"  # Optional
)
```

### List

```python
memories = client.memories.list(page=1, per_page=50)

for mem in memories.data:
    print(mem.text)

print(f"Total: {memories.pagination.total}")
```

### Get

```python
memory = client.memories.get("mem_abc123")
```

### Update

```python
memory = client.memories.update(
    "mem_abc123",
    text="Updated content",
    metadata={"updated": True}
)
```

### Delete

```python
client.memories.delete("mem_abc123")

# Delete all (requires confirmation)
client.memories.delete_all(confirm=True)
```

## Query Operations

### Basic Query

```python
results = client.query(
    context="What does the user do for work?",
    max_memories=10,
    relevance_threshold=0.5
)
```

### Query with Answer

```python
response = client.query_answer(
    context="What programming languages does the user know?"
)

print(response.answer)
print(f"Based on {len(response.memories_used)} memories")
```

### Extract Memories

```python
result = client.extract(
    content="User: Hi, I'm John from Google.",
    content_type="conversation",
    metadata={"source": "chat"}
)

print(f"Extracted {result.extracted_count} memories")
```

## Error Handling

```python
from memory_api import MemoryClient
from memory_api.exceptions import (
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ValidationError,
)

client = MemoryClient(api_key="...")

try:
    memory = client.memories.get("mem_xyz")
except NotFoundError:
    print("Memory not found")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except AuthenticationError:
    print("Invalid API key")
except ValidationError as e:
    print(f"Invalid request: {e.message}")
```

## Async Support

```python
import asyncio
from memory_api import AsyncMemoryClient

async def main():
    client = AsyncMemoryClient(api_key="mem_live_...")

    # All methods are async
    memory = await client.memories.create(text="Hello")
    results = await client.query(context="greeting")

    await client.close()

asyncio.run(main())
```
