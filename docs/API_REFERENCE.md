# Memory Systems - API Reference

**Version**: 1.0
**License**: [Your License]
**Support**: [Your Support Contact]

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [General Memory System](#general-memory-system)
4. [Code Memory System](#code-memory-system)
5. [Data Models](#data-models)
6. [Configuration](#configuration)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

---

## Installation

### Requirements

- Python 3.9 or higher
- An async LLM function (OpenAI, Anthropic, or custom)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```python
from memory_lib import MemorySystem, CodeMemorySystem
print("Memory systems imported successfully!")
```

---

## Quick Start

### General Memory System

```python
import asyncio
from memory_lib import MemorySystem

# Define your LLM function (example with OpenAI)
async def my_small_llm(prompt: str) -> str:
    # Your LLM API call here
    # Example: return await openai.ChatCompletion.create(...)
    pass

async def my_primary_llm(prompt: str) -> str:
    # Your primary LLM API call here
    pass

async def main():
    # Initialize system
    system = MemorySystem(small_model_fn=my_small_llm)

    # Add memories
    system.add_memory("Python uses dynamic typing")
    system.add_memory("FastAPI is a modern web framework")

    # Query
    response = await system.query(
        context="User asking about Python web development",
        task="Explain how to build a REST API",
        primary_model_fn=my_primary_llm
    )
    print(response)

asyncio.run(main())
```

### Code Memory System

```python
from memory_lib import CodeMemorySystem
from memory_lib.codebase import CodeContext

async def main():
    # Initialize
    system = CodeMemorySystem(small_model_fn=my_small_llm)

    # Index codebase
    system.index_repository("src/", exclude_patterns=["*/tests/*"])

    # Query with context
    context = CodeContext(
        user_query="How does authentication work?",
        current_file="api/auth.py"
    )

    response = await system.query(context, primary_model_fn=my_primary_llm)
    print(response)

asyncio.run(main())
```

---

## General Memory System

### MemorySystem

The main interface for general-purpose memory management.

#### Constructor

```python
MemorySystem(
    small_model_fn: Callable[[str], Awaitable[str]],
    db_path: str = "memories.db",
    relevance_threshold: float = 0.7,
    max_memories: int = 10,
    batch_size: int = 10,
    retry_attempts: int = 3,
    retry_delay: float = 1.0
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `small_model_fn` | async function | Required | Function to call small LLM for scoring |
| `db_path` | str | `"memories.db"` | Path to SQLite database |
| `relevance_threshold` | float | `0.7` | Minimum relevance score (0.0-1.0) |
| `max_memories` | int | `10` | Maximum memories to return |
| `batch_size` | int | `10` | Concurrent API calls for scoring |
| `retry_attempts` | int | `3` | Number of retries on API failure |
| `retry_delay` | float | `1.0` | Seconds between retries |

**Example**:
```python
system = MemorySystem(
    small_model_fn=my_gpt_turbo_function,
    db_path="app_memories.db",
    relevance_threshold=0.75,
    max_memories=15
)
```

#### Methods

##### `add_memory`

Add a new memory to the system.

```python
add_memory(
    text: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

**Parameters**:
- `text`: Memory content (required, non-empty)
- `metadata`: Optional dictionary of metadata

**Returns**: Memory ID (string)

**Raises**:
- `ValueError`: If text is empty or None

**Example**:
```python
memory_id = system.add_memory(
    "Python type hints improve code clarity",
    metadata={"topic": "python", "subtopic": "typing"}
)
```

---

##### `get_memory`

Retrieve a specific memory by ID.

```python
get_memory(memory_id: str) -> Optional[Dict[str, Any]]
```

**Parameters**:
- `memory_id`: ID of memory to retrieve

**Returns**: Dictionary with keys `{id, text, metadata, timestamp}` or `None`

**Example**:
```python
memory = system.get_memory(memory_id)
if memory:
    print(f"Memory: {memory['text']}")
```

---

##### `update_memory`

Update an existing memory.

```python
update_memory(
    memory_id: str,
    text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool
```

**Parameters**:
- `memory_id`: ID of memory to update
- `text`: New text (optional, None = don't update)
- `metadata`: New metadata (optional, None = don't update)

**Returns**: `True` if updated, `False` if memory not found

**Example**:
```python
success = system.update_memory(
    memory_id,
    text="Updated content",
    metadata={"updated": True}
)
```

---

##### `delete_memory`

Delete a memory by ID.

```python
delete_memory(memory_id: str) -> bool
```

**Parameters**:
- `memory_id`: ID of memory to delete

**Returns**: `True` if deleted, `False` if not found

**Example**:
```python
if system.delete_memory(memory_id):
    print("Memory deleted")
```

---

##### `retrieve_relevant_memories`

Retrieve memories relevant to a context (async).

```python
async retrieve_relevant_memories(
    context: str
) -> List[ScoredMemory]
```

**Parameters**:
- `context`: Query context describing what you're looking for

**Returns**: List of `ScoredMemory` objects sorted by relevance (highest first)

**Example**:
```python
relevant = await system.retrieve_relevant_memories(
    "Explain Python web frameworks"
)

for mem in relevant:
    print(f"Score: {mem.relevance_score:.2f}")
    print(f"Text: {mem.text}")
    print(f"Reasoning: {mem.reasoning}")
```

---

##### `format_memories_for_prompt`

Format memories as text for LLM prompts.

```python
format_memories_for_prompt(
    memories: List[ScoredMemory],
    include_scores: bool = True
) -> str
```

**Parameters**:
- `memories`: List of ScoredMemory objects
- `include_scores`: Whether to include relevance scores

**Returns**: Formatted string

**Example**:
```python
formatted = system.format_memories_for_prompt(relevant)
prompt = f"Context:\n{formatted}\n\nTask: {user_task}"
```

---

##### `query`

Complete query pipeline: retrieve memories and call primary LLM (async).

```python
async query(
    context: str,
    task: str,
    primary_model_fn: Callable[[str], Awaitable[str]]
) -> str
```

**Parameters**:
- `context`: Context for memory retrieval
- `task`: Actual task for the primary LLM
- `primary_model_fn`: Async function to call primary LLM

**Returns**: Primary LLM response string

**Example**:
```python
response = await system.query(
    context="User asking about Python",
    task="Explain Python's key features in simple terms",
    primary_model_fn=my_gpt4_function
)
```

---

##### `update_retrieval_config`

Update retrieval configuration at runtime.

```python
update_retrieval_config(**kwargs)
```

**Parameters** (all optional):
- `relevance_threshold`: New threshold (0.0-1.0)
- `max_memories`: New maximum count
- `batch_size`: New batch size
- `retry_attempts`: New retry count
- `retry_delay`: New retry delay

**Example**:
```python
system.update_retrieval_config(
    relevance_threshold=0.85,
    max_memories=5
)
```

---

##### `get_stats`

Get system statistics.

```python
get_stats() -> Dict[str, Any]
```

**Returns**: Dictionary with:
- `total_memories`: Total memory count
- `relevance_threshold`: Current threshold
- `max_memories`: Current max
- `batch_size`: Current batch size

**Example**:
```python
stats = system.get_stats()
print(f"System contains {stats['total_memories']} memories")
```

---

##### `close`

Close database connection.

```python
close()
```

**Example**:
```python
system.close()
```

---

## Code Memory System

### CodeMemorySystem

Specialized system for code intelligence with automatic indexing.

#### Constructor

```python
CodeMemorySystem(
    small_model_fn: Callable[[str], Awaitable[str]],
    db_path: str = "code_memories.db",
    relevance_threshold: float = 0.7,
    max_memories: int = 15,
    batch_size: int = 10,
    enable_caching: bool = True,
    enable_dependency_boost: bool = True,
    enable_recency_boost: bool = True,
    dependency_boost_amount: float = 0.15,
    recency_boost_amount: float = 0.10,
    recency_days: int = 7,
    retry_attempts: int = 3,
    retry_delay: float = 1.0
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `small_model_fn` | async function | Required | Small LLM for scoring |
| `db_path` | str | `"code_memories.db"` | Database path |
| `relevance_threshold` | float | `0.7` | Min relevance score |
| `max_memories` | int | `15` | Max memories to return |
| `batch_size` | int | `10` | Concurrent API calls |
| `enable_caching` | bool | `True` | Enable score caching |
| `enable_dependency_boost` | bool | `True` | Boost dependent functions |
| `enable_recency_boost` | bool | `True` | Boost recently modified |
| `dependency_boost_amount` | float | `0.15` | Dependency score boost |
| `recency_boost_amount` | float | `0.10` | Recency score boost |
| `recency_days` | int | `7` | Days to consider recent |
| `retry_attempts` | int | `3` | API retry count |
| `retry_delay` | float | `1.0` | Retry delay (seconds) |

**Example**:
```python
system = CodeMemorySystem(
    small_model_fn=my_haiku_function,
    db_path="my_code.db",
    max_memories=20,
    enable_caching=True
)
```

#### Methods

##### `index_file`

Index a single source file.

```python
index_file(file_path: str) -> List[str]
```

**Parameters**:
- `file_path`: Path to source file

**Returns**: List of created memory IDs

**Raises**:
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If unsupported file type

**Supported Languages**: Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, Ruby, PHP, C#

**Example**:
```python
memory_ids = system.index_file("src/utils.py")
print(f"Indexed {len(memory_ids)} functions/classes")
```

---

##### `index_repository`

Index an entire directory recursively.

```python
index_repository(
    directory: str,
    exclude_patterns: Optional[List[str]] = None,
    recursive: bool = True
) -> List[str]
```

**Parameters**:
- `directory`: Root directory to index
- `exclude_patterns`: Glob patterns to exclude (e.g., `["*/tests/*", "*/__pycache__/*"]`)
- `recursive`: Whether to recurse into subdirectories

**Returns**: List of all created memory IDs

**Example**:
```python
memory_ids = system.index_repository(
    "src",
    exclude_patterns=[
        "*/tests/*",
        "*/__pycache__/*",
        "*/.git/*",
        "*.pyc"
    ],
    recursive=True
)
print(f"Indexed {len(memory_ids)} code entities")
```

---

##### `reindex_file`

Re-index a file (removes old entities, indexes anew).

```python
reindex_file(file_path: str) -> List[str]
```

**Parameters**:
- `file_path`: File to re-index

**Returns**: List of new memory IDs

**Use Case**: Call after modifying a file

**Example**:
```python
# File was edited
new_ids = system.reindex_file("src/auth.py")
```

---

##### `add_documentation_memory`

Add documentation memory.

```python
add_documentation_memory(
    title: str,
    content: str,
    category: str = "documentation",
    file_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

**Parameters**:
- `title`: Documentation title
- `content`: Documentation content
- `category`: Category (default "documentation")
- `file_path`: Related file path (optional)
- `metadata`: Additional metadata (optional)

**Returns**: Memory ID

**Example**:
```python
doc_id = system.add_documentation_memory(
    title="API Authentication",
    content="Our API uses JWT tokens for authentication...",
    file_path="src/auth.py",
    metadata={"section": "security"}
)
```

---

##### `add_debugging_session`

Add a debugging session memory.

```python
add_debugging_session(
    description: str,
    error: str,
    solution: str,
    file_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

**Parameters**:
- `description`: Problem description
- `error`: Error message or stack trace
- `solution`: How the problem was solved
- `file_path`: Related file (optional)
- `metadata`: Additional metadata (optional)

**Returns**: Memory ID

**Example**:
```python
debug_id = system.add_debugging_session(
    description="NoneType error in validate_token",
    error="AttributeError: 'NoneType' object has no attribute 'id'",
    solution="Added null check: if user is not None before accessing user.id",
    file_path="src/auth.py"
)
```

---

##### `retrieve_relevant_memories`

Retrieve code memories relevant to a context (async).

```python
async retrieve_relevant_memories(
    context: Union[str, CodeContext],
    include_file_hashes: bool = True
) -> List[ScoredMemory]
```

**Parameters**:
- `context`: String or CodeContext object
- `include_file_hashes`: Compute file hashes for caching

**Returns**: List of ScoredMemory objects

**Example**:
```python
from memory_lib.codebase import CodeContext

context = CodeContext(
    user_query="How does user authentication work?",
    current_file="src/api/auth.py",
    errors="AttributeError in validate_token function"
)

relevant = await system.retrieve_relevant_memories(context)
```

---

##### `query`

Complete query pipeline with code context (async).

```python
async query(
    context: Union[str, CodeContext],
    primary_model_fn: Callable[[str], Awaitable[str]],
    custom_prompt_template: Optional[str] = None
) -> str
```

**Parameters**:
- `context`: Query context (string or CodeContext)
- `primary_model_fn`: Async function for primary LLM
- `custom_prompt_template`: Custom template (optional)

**Returns**: Primary LLM response

**Example**:
```python
response = await system.query(
    context=CodeContext(
        user_query="Add error handling to the login function",
        current_file="src/auth.py"
    ),
    primary_model_fn=my_gpt4_function
)
```

---

##### `clear_cache`

Clear score cache (useful after major code changes).

```python
clear_cache()
```

**Example**:
```python
system.clear_cache()
```

---

##### `get_stats`

Get system statistics.

```python
get_stats() -> Dict[str, Any]
```

**Returns**: Dictionary with:
- `total_code_memories`: Code entity count
- `total_non_code_memories`: Documentation/debugging count
- `total_memories`: Total count
- `cache_stats`: Cache information
- Retrieval configuration

**Example**:
```python
stats = system.get_stats()
print(f"Code entities: {stats['total_code_memories']}")
print(f"Documentation: {stats['total_non_code_memories']}")
print(f"Cache size: {stats['cache_stats']['size']}")
```

---

##### `close`

Close database connection and cleanup.

```python
close()
```

**Example**:
```python
system.close()
```

---

## Data Models

### ScoredMemory

Represents a memory with its relevance score.

```python
@dataclass
class ScoredMemory:
    memory_id: str              # Unique identifier
    text: str                   # Memory content
    metadata: Optional[Dict]    # Metadata dictionary
    timestamp: str              # ISO 8601 timestamp
    relevance_score: float      # 0.0 to 1.0
    reasoning: str              # Why this score was assigned
```

**Example**:
```python
for memory in relevant_memories:
    print(f"ID: {memory.memory_id}")
    print(f"Score: {memory.relevance_score:.2f}")
    print(f"Text: {memory.text}")
    print(f"Reasoning: {memory.reasoning}")
    print(f"Metadata: {memory.metadata}")
    print("---")
```

---

### CodeContext

Rich context for code-related queries.

```python
@dataclass
class CodeContext:
    user_query: str                              # Required: the question/task
    current_file: Optional[str] = None          # File being worked on
    recent_changes: Optional[str] = None        # Recent code changes
    errors: Optional[str] = None                # Error messages
    accessed_files: Optional[List[str]] = None  # Recently viewed files
    additional_context: Optional[str] = None    # Any other context
```

**Example**:
```python
from memory_lib.codebase import CodeContext

context = CodeContext(
    user_query="Debug the authentication issue",
    current_file="src/api/auth.py",
    errors="AttributeError: 'NoneType' object has no attribute 'id'",
    recent_changes="Modified validate_token to check user roles",
    accessed_files=["src/api/auth.py", "src/models/user.py"]
)

response = await system.query(context, primary_model_fn)
```

---

## Configuration

### Default Configuration

**General Memory System**:
```python
{
    "relevance_threshold": 0.7,
    "max_memories": 10,
    "batch_size": 10,
    "retry_attempts": 3,
    "retry_delay": 1.0
}
```

**Code Memory System**:
```python
{
    "relevance_threshold": 0.7,
    "max_memories": 15,
    "batch_size": 10,
    "enable_caching": True,
    "enable_dependency_boost": True,
    "enable_recency_boost": True,
    "dependency_boost_amount": 0.15,
    "recency_boost_amount": 0.10,
    "recency_days": 7,
    "retry_attempts": 3,
    "retry_delay": 1.0
}
```

### Tuning Guidelines

**Relevance Threshold** (0.0 - 1.0):
- `0.5-0.6`: Very permissive, more context
- `0.7`: Balanced (default)
- `0.8-0.9`: Strict, only highly relevant

**Max Memories**:
- `5-10`: Small context, focused
- `10-20`: Medium context (recommended)
- `20-50`: Large context, comprehensive

**Batch Size**:
- `5`: Conservative (for rate-limited APIs)
- `10`: Balanced (default)
- `20-50`: Aggressive (for high rate limits)

**Boost Amounts** (Code System):
- `0.05-0.10`: Subtle boost
- `0.15`: Moderate (default)
- `0.20-0.30`: Strong boost

**Example Tuning**:
```python
# High precision, low recall
system.update_retrieval_config(
    relevance_threshold=0.9,
    max_memories=5
)

# High recall, moderate precision
system.update_retrieval_config(
    relevance_threshold=0.6,
    max_memories=30
)
```

---

## Error Handling

### Common Exceptions

**ValueError**:
```python
try:
    system.add_memory("")  # Empty text
except ValueError as e:
    print(f"Invalid input: {e}")
```

**FileNotFoundError**:
```python
try:
    system.index_file("nonexistent.py")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

**API Errors** (from LLM function):
```python
async def safe_llm_call(prompt: str) -> str:
    try:
        return await my_llm_api(prompt)
    except APIError as e:
        # Log error
        raise  # Re-raise for retry logic
```

### Retry Logic

The system automatically retries failed API calls:

```python
system = MemorySystem(
    small_model_fn=my_llm,
    retry_attempts=5,      # Try up to 5 times
    retry_delay=2.0        # Wait 2 seconds between retries
)
```

Retry uses exponential backoff:
- Attempt 1: Immediate
- Attempt 2: Wait retry_delay
- Attempt 3: Wait retry_delay * 2
- Attempt 4: Wait retry_delay * 4
- ...

---

## Best Practices

### 1. LLM Function Design

**Good**:
```python
async def my_small_llm(prompt: str) -> str:
    """Call GPT-3.5-turbo for scoring."""
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # Deterministic for scoring
        max_tokens=100    # Short responses
    )
    return response.choices[0].message.content
```

**Why**:
- Low temperature for consistent scoring
- Limited tokens (scoring responses are short)
- Proper error handling
- Cost-effective model choice

---

### 2. Metadata Design

**Good**:
```python
system.add_memory(
    "FastAPI uses Pydantic for data validation",
    metadata={
        "topic": "web_frameworks",
        "language": "python",
        "subtopic": "validation",
        "version": "0.100.0",
        "importance": "high"
    }
)
```

**Why**:
- Structured, consistent keys
- Useful for filtering and analysis
- Hierarchical organization

---

### 3. Context Design

**Good**:
```python
context = CodeContext(
    user_query="Fix authentication bug in login endpoint",
    current_file="src/api/auth.py",
    errors="AttributeError: 'NoneType' object has no attribute 'id'",
    recent_changes="Added role-based access control",
    accessed_files=["src/api/auth.py", "src/models/user.py"]
)
```

**Why**:
- Specific, detailed query
- Relevant contextual information
- Helps retrieval find right memories

---

### 4. Indexing Strategy

**Good**:
```python
# Index incrementally
system.index_repository("src/core")
system.index_repository("src/api")
system.index_repository("src/utils")

# Exclude unnecessary files
exclude_patterns = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/.git/*",
    "*.pyc",
    "*/migrations/*",
    "*/node_modules/*"
]
```

**Why**:
- Incremental indexing allows progress tracking
- Excluding tests/build artifacts reduces noise
- Focusing on source code improves relevance

---

### 5. Cache Management

**Good**:
```python
# After major refactoring
system.clear_cache()

# After modifying a file
system.reindex_file("src/auth.py")

# Periodic cleanup
if stats["cache_stats"]["size"] > 10000:
    system.clear_cache()
```

**Why**:
- Cache invalidation prevents stale scores
- Reindexing updates code entities
- Prevents unbounded cache growth

---

### 6. Resource Cleanup

**Good**:
```python
system = MemorySystem(small_model_fn=my_llm)
try:
    # Use system
    await system.query(...)
finally:
    system.close()

# Or use context manager (if supported)
with MemorySystem(small_model_fn=my_llm) as system:
    await system.query(...)
```

**Why**:
- Ensures database connections are closed
- Prevents resource leaks
- Clean shutdown

---

## Examples

### Example 1: Customer Support Bot

```python
import asyncio
from memory_lib import MemorySystem

async def support_bot():
    system = MemorySystem(small_model_fn=gpt_turbo_fn)

    # Load company knowledge
    system.add_memory(
        "Returns are accepted within 30 days of purchase",
        metadata={"category": "returns", "policy": True}
    )
    system.add_memory(
        "Free shipping on orders over $50",
        metadata={"category": "shipping", "policy": True}
    )

    # Customer query
    response = await system.query(
        context="Customer asking about returns",
        task="Explain our return policy",
        primary_model_fn=gpt4_fn
    )

    print(response)
    system.close()

asyncio.run(support_bot())
```

---

### Example 2: Code Assistant

```python
from memory_lib import CodeMemorySystem
from memory_lib.codebase import CodeContext

async def code_assistant():
    system = CodeMemorySystem(small_model_fn=haiku_fn)

    # Index codebase
    print("Indexing repository...")
    ids = system.index_repository(
        "src",
        exclude_patterns=["*/tests/*", "*/__pycache__/*"]
    )
    print(f"Indexed {len(ids)} code entities")

    # Add documentation
    system.add_documentation_memory(
        title="Authentication Flow",
        content="Users authenticate via JWT tokens. Tokens expire after 24h.",
        category="architecture"
    )

    # Query
    context = CodeContext(
        user_query="How do I add a new API endpoint?",
        current_file="src/api/routes.py"
    )

    response = await system.query(context, primary_model_fn=gpt4_fn)
    print(response)

    system.close()

asyncio.run(code_assistant())
```

---

### Example 3: Research Assistant

```python
from memory_lib import MemorySystem

async def research_assistant():
    system = MemorySystem(
        small_model_fn=haiku_fn,
        max_memories=20  # More context for research
    )

    # Add research papers
    papers = [
        ("Transformer architecture uses self-attention", {"paper": "Attention Is All You Need"}),
        ("BERT uses bidirectional training", {"paper": "BERT"}),
        ("GPT-3 has 175B parameters", {"paper": "GPT-3"})
    ]

    for text, metadata in papers:
        system.add_memory(text, metadata=metadata)

    # Research query
    response = await system.query(
        context="Researching transformer models",
        task="Compare BERT and GPT architectures",
        primary_model_fn=gpt4_fn
    )

    print(response)
    system.close()

asyncio.run(research_assistant())
```

---

## Support

For issues, questions, or feature requests:

- **Documentation**: [Link to docs]
- **GitHub Issues**: [Link to issues]
- **Email Support**: [Your email]
- **Community**: [Link to forum/Discord]

---

## Changelog

### Version 1.0
- Initial release
- General memory system
- Code memory system
- Caching and optimization features

---

**Next Steps**:
- See [Quick Start Guide](QUICKSTART.md) for step-by-step tutorials
- See [Architecture Documentation](01_ARCHITECTURE.md) for system design
- See [Component Documentation](02_COMPONENTS.md) for detailed specifications
