# Memory System API Reference

## General Memory System

### MemorySystem

Main class for general memory management.

```python
from memory_lib import MemorySystem

system = MemorySystem(
    small_model_fn=your_small_model_function,
    db_path="memories.db",
    relevance_threshold=0.7,
    max_memories=10
)
```

#### Storage Methods

##### `add_memory(text, metadata=None, memory_id=None) -> str`

Add a new memory to the system.

**Parameters:**
- `text` (str): The content of the memory
- `metadata` (dict, optional): Additional metadata
- `memory_id` (str, optional): Custom ID (UUID generated if not provided)

**Returns:** Memory ID (str)

**Example:**
```python
mid = system.add_memory(
    "Python is dynamically typed",
    metadata={"topic": "python", "category": "basics"}
)
```

##### `get_memory(memory_id) -> dict | None`

Retrieve a specific memory by ID.

**Returns:** Memory dictionary or None

##### `get_all_memories() -> list[dict]`

Get all memories, ordered by timestamp (most recent first).

##### `update_memory(memory_id, text=None, metadata=None) -> bool`

Update an existing memory.

**Returns:** True if updated, False if not found

##### `delete_memory(memory_id) -> bool`

Delete a memory.

**Returns:** True if deleted, False if not found

##### `count_memories() -> int`

Get total number of memories.

##### `clear_all_memories() -> int`

Delete all memories (irreversible).

**Returns:** Number of memories deleted

#### Retrieval Methods

##### `async retrieve_relevant_memories(context) -> list[ScoredMemory]`

Retrieve relevant memories for a given context using LLM-based scoring.

**Parameters:**
- `context` (str): The current context or query

**Returns:** List of ScoredMemory objects sorted by relevance

**Example:**
```python
memories = await system.retrieve_relevant_memories(
    "Tell me about Python programming"
)
```

##### `format_memories_for_prompt(memories, include_scores=True) -> str`

Format memories for inclusion in an LLM prompt.

**Returns:** Formatted string with XML-style tags

##### `async query(context, task, primary_model_fn, include_scores=True) -> str`

Complete query pipeline: retrieve memories and call primary model.

**Parameters:**
- `context` (str): Current context
- `task` (str): Specific task for the primary model
- `primary_model_fn` (callable): Async function to call primary LLM
- `include_scores` (bool): Include relevance scores in prompt

**Returns:** Response from primary model

**Example:**
```python
response = await system.query(
    context="User asking about web frameworks",
    task="Compare FastAPI and Flask",
    primary_model_fn=my_gpt4_function
)
```

##### `async query_with_custom_prompt(context, prompt_template, primary_model_fn) -> str`

Query with a custom prompt template.

**Parameters:**
- `prompt_template` (str): Template with `{memories}` placeholder

#### Configuration Methods

##### `update_retrieval_config(relevance_threshold=None, max_memories=None, batch_size=None)`

Update retrieval configuration dynamically.

**Example:**
```python
system.update_retrieval_config(
    relevance_threshold=0.8,
    max_memories=15
)
```

##### `get_stats() -> dict`

Get system statistics.

**Returns:** Dictionary with counts and configuration

---

## Code Memory System

### CodeMemorySystem

Extended system for code intelligence.

```python
from memory_lib import CodeMemorySystem
from memory_lib.codebase import CodeContext

system = CodeMemorySystem(
    small_model_fn=your_small_model_function,
    db_path="code_memories.db",
    relevance_threshold=0.7,
    max_memories=15,
    enable_caching=True,
    enable_dependency_boost=True,
    enable_recency_boost=True
)
```

#### Indexing Methods

##### `index_file(file_path, auto_store=True) -> list[dict]`

Index a single source file.

**Parameters:**
- `file_path` (str): Path to source file
- `auto_store` (bool): Automatically store extracted entities

**Returns:** List of extracted entity dictionaries

**Example:**
```python
entities = system.index_file("src/api/handlers.py")
```

##### `index_repository(directory, exclude_patterns=None, recursive=True, auto_store=True) -> list[dict]`

Index an entire codebase.

**Parameters:**
- `directory` (str): Repository directory path
- `exclude_patterns` (list, optional): Glob patterns to exclude
- `recursive` (bool): Recursively index subdirectories
- `auto_store` (bool): Automatically store entities

**Returns:** List of all extracted entities

**Example:**
```python
entities = system.index_repository(
    "src/",
    exclude_patterns=["*/tests/*", "*/__pycache__/*", "*.pyc"],
    recursive=True
)
```

##### `reindex_file(file_path) -> int`

Re-index a file after modifications (removes old memories first).

**Returns:** Number of new entities indexed

#### Storage Methods

##### `add_code_memory(**kwargs) -> str`

Add a code memory manually.

**Parameters:**
- `file_path` (str): Source file path
- `entity_name` (str, optional): Function/class name
- `code_snippet` (str, optional): Code content
- `docstring` (str, optional): Documentation
- `signature` (str, optional): Function signature
- `language` (str, optional): Programming language
- `dependencies` (list, optional): Referenced entities
- `imports` (list, optional): Import statements
- `complexity` (str, optional): "low", "medium", "high"
- `metadata` (dict, optional): Additional metadata

**Example:**
```python
mid = system.add_code_memory(
    file_path="api/auth.py",
    entity_name="authenticate",
    code_snippet="def authenticate(user): ...",
    language="python",
    complexity="medium"
)
```

##### `add_documentation_memory(title, content, category="documentation", file_path=None, metadata=None) -> str`

Add documentation (README, architecture, etc.).

**Example:**
```python
mid = system.add_documentation_memory(
    title="API Architecture",
    content="Our API uses...",
    category="architecture"
)
```

##### `add_debugging_session(title, content, metadata=None) -> str`

Add a debugging session memory.

**Example:**
```python
mid = system.add_debugging_session(
    title="Fixed race condition",
    content="Issue was in async handler...",
    metadata={"severity": "high"}
)
```

##### `get_memory(memory_id) -> dict | None`

Get a memory (code or non-code) by ID.

##### `get_all_memories() -> list[dict]`

Get all memories with 'type' field ('code' or 'non-code').

##### `get_memories_by_file(file_path) -> list[dict]`

Get all code memories for a specific file.

##### `delete_memory(memory_id) -> bool`

Delete a memory.

#### Retrieval Methods

##### `async retrieve_relevant_memories(context, file_hashes=None) -> list[ScoredMemory]`

Retrieve relevant memories for a code context.

**Parameters:**
- `context` (CodeContext): Rich code context object
- `file_hashes` (dict, optional): File path -> hash mapping for caching

**Returns:** List of ScoredMemory objects

**Example:**
```python
context = CodeContext(
    user_query="Fix authentication bug",
    current_file="api/auth.py",
    errors="AttributeError in validate_token"
)
memories = await system.retrieve_relevant_memories(context)
```

##### `format_memories_for_prompt(memories, group_by_file=True, include_scores=True) -> str`

Format code memories for prompt.

##### `async query(context, primary_model_fn, include_scores=True, file_hashes=None) -> str`

Complete query pipeline with code context.

**Example:**
```python
response = await system.query(
    context=code_context,
    primary_model_fn=my_gpt4_function
)
```

##### `async query_with_custom_prompt(context, prompt_template, primary_model_fn, ...) -> str`

Query with custom template. Template should include `{memories}` and `{context}` placeholders.

#### Configuration Methods

##### `update_retrieval_config(relevance_threshold=None, max_memories=None, batch_size=None)`

Update retrieval settings.

##### `get_stats() -> dict`

Get comprehensive statistics including cache info.

**Returns:**
```python
{
    "code_memories": int,
    "non_code_memories": int,
    "total_memories": int,
    "retrieval_config": {...},
    "cache": {
        "caching_enabled": bool,
        "cache_size": int,
        "dependency_boost_enabled": bool,
        "recency_boost_enabled": bool
    }
}
```

##### `clear_cache()`

Clear the relevance score cache.

##### `clear_all_memories() -> int`

Delete all memories (irreversible).

### CodeContext

Context object for code memory retrieval.

```python
from memory_lib.codebase import CodeContext

context = CodeContext(
    user_query="The task or question",
    current_file="path/to/file.py",
    recent_changes="Description of changes",
    errors="Error messages or stack traces",
    accessed_files=["file1.py", "file2.py"],
    additional_context="Any other relevant info"
)
```

**Required:**
- `user_query` (str): The task or question

**Optional:**
- `current_file` (str): File being edited
- `recent_changes` (str): Recent code changes
- `errors` (str): Active errors or exceptions
- `accessed_files` (list): Recently viewed files
- `additional_context` (str): Other context

**Methods:**
- `to_context_string() -> str`: Convert to formatted string

### CodeIndexer

Utility for extracting code entities.

```python
from memory_lib.codebase import CodeIndexer

indexer = CodeIndexer()
entities = indexer.index_file("src/api.py")
```

#### Methods

##### `index_file(file_path) -> list[dict]`

Index a single file.

##### `index_directory(directory, exclude_patterns=None, recursive=True) -> list[dict]`

Index a directory.

##### `compute_file_hash(file_path) -> str`

Compute SHA256 hash for change detection.

**Example:**
```python
hash1 = indexer.compute_file_hash("api.py")
# ... modify file ...
hash2 = indexer.compute_file_hash("api.py")
if hash1 != hash2:
    system.reindex_file("api.py")
```

### ScoredMemory

Dataclass representing a memory with relevance score.

**Attributes:**
- `memory_id` (str): Unique identifier
- `text` (str): Memory content
- `metadata` (dict): Associated metadata
- `timestamp` (str): ISO format timestamp
- `relevance_score` (float): Score from 0 to 1
- `reasoning` (str): Explanation of score

---

## Integration Examples

### OpenAI Integration

```python
import openai
from memory_lib import CodeMemorySystem

async def small_model_fn(prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

async def primary_model_fn(prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

system = CodeMemorySystem(small_model_fn=small_model_fn)
```

### Anthropic (Claude) Integration

```python
import anthropic
from memory_lib import CodeMemorySystem

client = anthropic.AsyncAnthropic(api_key="your-key")

async def small_model_fn(prompt: str) -> str:
    message = await client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

async def primary_model_fn(prompt: str) -> str:
    message = await client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

system = CodeMemorySystem(small_model_fn=small_model_fn)
```

---

## Error Handling

All methods handle errors gracefully:

- File operations return empty lists on failure
- Missing memories return `None`
- API errors are retried with exponential backoff
- Failed scoring assigns score of 0.0

Always check return values:

```python
memory = system.get_memory(mem_id)
if memory is None:
    print("Memory not found")

success = system.delete_memory(mem_id)
if not success:
    print("Memory not found or already deleted")
```

---

## Performance Tuning

### Batch Size

Controls parallel API calls:

```python
system = MemorySystem(
    small_model_fn=...,
    batch_size=20  # Higher = faster but may hit rate limits
)
```

### Relevance Threshold

Controls quality vs quantity:

```python
# More selective (fewer, higher quality memories)
system.update_retrieval_config(relevance_threshold=0.8)

# More inclusive (more memories, varied quality)
system.update_retrieval_config(relevance_threshold=0.6)
```

### Max Memories

Controls context window usage:

```python
# More context (may hit token limits)
system.update_retrieval_config(max_memories=25)

# Less context (faster, cheaper)
system.update_retrieval_config(max_memories=5)
```

### Caching (Code System)

```python
# Enable/disable at initialization
system = CodeMemorySystem(
    enable_caching=True,  # Recommended
    enable_dependency_boost=True,
    enable_recency_boost=True
)

# Clear cache when needed
system.clear_cache()
```
