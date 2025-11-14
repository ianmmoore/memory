# Memory Systems - Component Documentation

## Table of Contents

1. [General Memory System Components](#general-memory-system-components)
   - [MemoryStorage](#memorystorage)
   - [MemoryRetrieval](#memoryretrieval)
   - [MemorySystem](#memorysystem)
2. [Code Memory System Components](#code-memory-system-components)
   - [CodeIndexer](#codeindexer)
   - [CodeMemoryStorage](#codememorystorage)
   - [CodeMemoryRetrieval](#codememoryretrieval)
   - [CodeMemorySystem](#codememorysystem)
3. [Data Models](#data-models)
4. [Configuration](#configuration)

---

## General Memory System Components

### MemoryStorage

**Location**: `memory_lib/general/storage.py`

**Purpose**: Provides persistent SQLite storage for general-purpose memories with CRUD operations.

#### Responsibilities

- Create and manage SQLite database connection
- Store memories with text, metadata, and timestamps
- Provide create, read, update, delete operations
- Generate unique IDs for memories
- Track memory count and statistics

#### Database Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,           -- UUID4 string
    text TEXT NOT NULL,            -- Memory content
    metadata TEXT,                 -- JSON-encoded dict
    timestamp TEXT NOT NULL        -- ISO 8601 format (UTC)
);

CREATE INDEX idx_timestamp ON memories(timestamp);
```

#### Public Methods

##### `__init__(self, db_path: str = "memories.db")`
Initialize storage with database file path.

**Parameters**:
- `db_path`: Path to SQLite database file (created if doesn't exist)

**Example**:
```python
storage = MemoryStorage("my_memories.db")
```

##### `add_memory(self, text: str, metadata: Optional[Dict] = None, memory_id: Optional[str] = None) -> str`
Add a new memory to storage.

**Parameters**:
- `text`: Memory content (required, non-empty)
- `metadata`: Optional dictionary of metadata
- `memory_id`: Optional custom ID (generated if not provided)

**Returns**: Memory ID (string)

**Raises**:
- `ValueError`: If text is empty or None

**Example**:
```python
memory_id = storage.add_memory(
    "Python uses duck typing",
    metadata={"language": "python", "topic": "typing"}
)
```

##### `get_memory(self, memory_id: str) -> Optional[Dict]`
Retrieve a single memory by ID.

**Parameters**:
- `memory_id`: Memory ID to retrieve

**Returns**: Dictionary with keys `{id, text, metadata, timestamp}` or None if not found

**Example**:
```python
memory = storage.get_memory(memory_id)
if memory:
    print(memory["text"])
```

##### `get_all_memories(self) -> List[Dict]`
Retrieve all memories in database.

**Returns**: List of memory dictionaries, ordered by timestamp (newest first)

**Example**:
```python
all_memories = storage.get_all_memories()
print(f"Total memories: {len(all_memories)}")
```

##### `update_memory(self, memory_id: str, text: Optional[str] = None, metadata: Optional[Dict] = None) -> bool`
Update an existing memory's text and/or metadata.

**Parameters**:
- `memory_id`: ID of memory to update
- `text`: New text (None = don't update)
- `metadata`: New metadata (None = don't update)

**Returns**: True if updated, False if memory not found

**Example**:
```python
success = storage.update_memory(
    memory_id,
    text="Python uses dynamic typing",
    metadata={"language": "python", "topic": "typing", "updated": True}
)
```

##### `delete_memory(self, memory_id: str) -> bool`
Delete a memory by ID.

**Parameters**:
- `memory_id`: ID of memory to delete

**Returns**: True if deleted, False if not found

**Example**:
```python
if storage.delete_memory(memory_id):
    print("Memory deleted successfully")
```

##### `count_memories(self) -> int`
Count total memories in database.

**Returns**: Integer count

**Example**:
```python
count = storage.count_memories()
print(f"Database contains {count} memories")
```

##### `clear_all(self) -> int`
Delete all memories from database.

**Returns**: Number of memories deleted

**Example**:
```python
deleted = storage.clear_all()
print(f"Cleared {deleted} memories")
```

##### `close(self)`
Close database connection.

**Example**:
```python
storage.close()
```

#### Context Manager Support

```python
with MemoryStorage("memories.db") as storage:
    storage.add_memory("Example memory")
# Database automatically closed
```

---

### MemoryRetrieval

**Location**: `memory_lib/general/retrieval.py`

**Purpose**: LLM-based relevance scoring and intelligent memory selection.

#### Responsibilities

- Score all memories for relevance using small LLM
- Batch processing for efficiency
- Filter memories by relevance threshold
- Select top-K most relevant memories
- Format memories for inclusion in prompts
- Handle API errors with retries

#### Data Model: ScoredMemory

```python
@dataclass
class ScoredMemory:
    memory_id: str
    text: str
    metadata: Optional[Dict]
    timestamp: str
    relevance_score: float      # 0.0 to 1.0
    reasoning: str              # Why this score was assigned
```

#### Public Methods

##### `__init__(self, small_model_fn, relevance_threshold=0.7, max_memories=10, batch_size=10, retry_attempts=3, retry_delay=1.0)`
Initialize retrieval system.

**Parameters**:
- `small_model_fn`: Async function `(prompt: str) -> str` for LLM scoring
- `relevance_threshold`: Minimum score to include (0.0-1.0, default 0.7)
- `max_memories`: Maximum memories to return (default 10)
- `batch_size`: Number of concurrent scoring calls (default 10)
- `retry_attempts`: Number of retries on API failure (default 3)
- `retry_delay`: Delay between retries in seconds (default 1.0)

**Example**:
```python
async def my_llm(prompt: str) -> str:
    # Call your LLM API
    return response_text

retrieval = MemoryRetrieval(
    small_model_fn=my_llm,
    relevance_threshold=0.8,
    max_memories=5
)
```

##### `async retrieve_relevant_memories(self, context: str, memories: List[Dict]) -> List[ScoredMemory]`
Main retrieval method: score, filter, and select memories.

**Parameters**:
- `context`: Query context (what the user is asking about)
- `memories`: List of memory dicts from storage

**Returns**: List of ScoredMemory objects, sorted by relevance (highest first)

**Example**:
```python
memories = storage.get_all_memories()
relevant = await retrieval.retrieve_relevant_memories(
    "Explain Python typing",
    memories
)
for mem in relevant:
    print(f"Score: {mem.relevance_score:.2f} - {mem.text}")
```

##### `async score_all_memories(self, context: str, memories: List[Dict]) -> List[ScoredMemory]`
Score all memories in batches.

**Parameters**:
- `context`: Query context
- `memories`: List of memory dicts

**Returns**: List of ScoredMemory objects with scores

**Note**: Usually called internally by `retrieve_relevant_memories`

##### `filter_and_select(self, scored_memories: List[ScoredMemory]) -> List[ScoredMemory]`
Filter by threshold and select top-K.

**Parameters**:
- `scored_memories`: List of scored memories

**Returns**: Filtered and sorted list (max length = max_memories)

**Example**:
```python
scored = await retrieval.score_all_memories(context, memories)
selected = retrieval.filter_and_select(scored)
```

##### `format_memories_for_prompt(self, memories: List[ScoredMemory], include_scores: bool = True) -> str`
Format memories as text for inclusion in LLM prompt.

**Parameters**:
- `memories`: List of ScoredMemory objects
- `include_scores`: Whether to include relevance scores (default True)

**Returns**: Formatted string

**Example**:
```python
formatted = retrieval.format_memories_for_prompt(relevant)
prompt = f"Context:\n{formatted}\n\nTask: {user_task}"
```

**Output format**:
```
Memory 1 (Relevance: 0.95):
Python uses dynamic typing
[Metadata: language=python, topic=typing]

Memory 2 (Relevance: 0.82):
...
```

##### `update_config(self, **kwargs)`
Update configuration parameters at runtime.

**Parameters**:
- `relevance_threshold`: New threshold (optional)
- `max_memories`: New maximum (optional)
- `batch_size`: New batch size (optional)
- `retry_attempts`: New retry count (optional)
- `retry_delay`: New retry delay (optional)

**Example**:
```python
retrieval.update_config(
    relevance_threshold=0.9,
    max_memories=3
)
```

#### Scoring Algorithm

The scoring process for each memory:

1. **Create Prompt**:
```
Context: {user_context}

Memory: {memory_text}
Metadata: {metadata_string}

Is this memory relevant to the context? Rate its relevance from 0 to 1.
Score: <number 0-1>
Reason: <brief explanation>
```

2. **Call LLM**: Send prompt to small model function

3. **Parse Response**: Extract score and reasoning using regex
   - Pattern: `Score:\s*([\d.]+)`
   - Clamp score to [0.0, 1.0]

4. **Retry on Failure**: Up to `retry_attempts` times with exponential backoff

5. **Return ScoredMemory**: Package results in data class

---

### MemorySystem

**Location**: `memory_lib/general/memory_system.py`

**Purpose**: High-level facade that combines storage and retrieval for easy use.

#### Responsibilities

- Initialize and configure storage and retrieval components
- Provide unified API for memory operations
- Orchestrate complete query pipeline
- Manage configuration updates
- Provide statistics and monitoring

#### Public Methods

##### `__init__(self, small_model_fn, db_path="memories.db", **retrieval_kwargs)`
Initialize complete memory system.

**Parameters**:
- `small_model_fn`: Async LLM function for scoring
- `db_path`: Database path (default "memories.db")
- `**retrieval_kwargs`: Any parameters for MemoryRetrieval

**Example**:
```python
system = MemorySystem(
    small_model_fn=my_llm_function,
    db_path="app_memories.db",
    relevance_threshold=0.75,
    max_memories=15
)
```

##### `add_memory(self, text: str, metadata: Optional[Dict] = None) -> str`
Add a memory (delegates to storage).

**Parameters**: Same as `MemoryStorage.add_memory`

**Returns**: Memory ID

**Example**:
```python
memory_id = system.add_memory(
    "FastAPI is a modern Python web framework",
    metadata={"topic": "web", "language": "python"}
)
```

##### `get_memory(self, memory_id: str) -> Optional[Dict]`
Get a specific memory (delegates to storage).

##### `update_memory(self, memory_id: str, text: Optional[str] = None, metadata: Optional[Dict] = None) -> bool`
Update a memory (delegates to storage).

##### `delete_memory(self, memory_id: str) -> bool`
Delete a memory (delegates to storage).

##### `async retrieve_relevant_memories(self, context: str) -> List[ScoredMemory]`
Retrieve relevant memories for a context.

**Parameters**:
- `context`: Query context

**Returns**: List of relevant ScoredMemory objects

**Example**:
```python
relevant = await system.retrieve_relevant_memories(
    "How do I build a REST API?"
)
```

##### `format_memories_for_prompt(self, memories: List[ScoredMemory], include_scores: bool = True) -> str`
Format memories for prompt inclusion (delegates to retrieval).

##### `async query(self, context: str, task: str, primary_model_fn) -> str`
Complete query pipeline: retrieve memories and call primary LLM.

**Parameters**:
- `context`: Query context for retrieval
- `task`: Actual task for primary LLM
- `primary_model_fn`: Async function for main LLM call

**Returns**: Primary LLM response string

**Example**:
```python
response = await system.query(
    context="User asking about web frameworks",
    task="Explain how to build a REST API with FastAPI",
    primary_model_fn=my_gpt4_function
)
print(response)
```

**Internal Flow**:
1. Retrieve relevant memories using context
2. Format memories as string
3. Build prompt: `Relevant context:\n{memories}\n\nTask: {task}`
4. Call primary_model_fn with prompt
5. Return response

##### `update_retrieval_config(self, **kwargs)`
Update retrieval configuration (delegates to retrieval).

**Example**:
```python
system.update_retrieval_config(
    relevance_threshold=0.85,
    max_memories=20
)
```

##### `get_stats(self) -> Dict`
Get system statistics.

**Returns**: Dict with keys:
- `total_memories`: Total count
- `relevance_threshold`: Current threshold
- `max_memories`: Current max
- `batch_size`: Current batch size

**Example**:
```python
stats = system.get_stats()
print(f"System has {stats['total_memories']} memories")
```

##### `close(self)`
Close database connection.

---

## Code Memory System Components

### CodeIndexer

**Location**: `memory_lib/codebase/indexer.py`

**Purpose**: Extract code entities (functions, classes) from source files.

#### Responsibilities

- Parse source files using language-specific strategies
- Extract functions with signatures, docstrings, and dependencies
- Extract classes with methods
- Compute file hashes for change detection
- Support multiple programming languages
- Estimate code complexity

#### Supported Languages

- **Python** (AST-based): Full parsing with dependencies
- **JavaScript/TypeScript** (regex-based): Function and class extraction
- **Java, C/C++, Go, Rust, Ruby, PHP, C#** (regex-based): Basic extraction

#### Public Methods

##### `__init__(self)`
Initialize code indexer.

**Example**:
```python
indexer = CodeIndexer()
```

##### `index_file(self, file_path: str) -> List[Dict]`
Index a single source file.

**Parameters**:
- `file_path`: Path to source file

**Returns**: List of entity dictionaries

**Raises**:
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If unsupported file type

**Example**:
```python
entities = indexer.index_file("src/utils.py")
for entity in entities:
    print(f"Found {entity['entity_name']} in {entity['file_path']}")
```

**Entity Structure**:
```python
{
    "file_path": "src/utils.py",
    "entity_name": "calculate_sum",
    "code_snippet": "def calculate_sum(a, b):\n    return a + b",
    "docstring": "Calculate the sum of two numbers",
    "signature": "calculate_sum(a, b)",
    "language": "python",
    "dependencies": ["print", "log"],  # Functions called
    "imports": ["import math", "from typing import List"],
    "complexity": "low",  # low | medium | high
    "last_modified": "2024-01-15T10:30:00Z"
}
```

##### `index_directory(self, directory: str, exclude_patterns: Optional[List[str]] = None, recursive: bool = True) -> List[Dict]`
Index all supported files in a directory.

**Parameters**:
- `directory`: Path to directory
- `exclude_patterns`: Glob patterns to exclude (e.g., `["*/tests/*", "*/__pycache__/*"]`)
- `recursive`: Whether to recurse into subdirectories (default True)

**Returns**: List of all extracted entities

**Example**:
```python
entities = indexer.index_directory(
    "src",
    exclude_patterns=["*/test_*.py", "*/.git/*"],
    recursive=True
)
print(f"Indexed {len(entities)} entities")
```

##### `compute_file_hash(self, file_path: str) -> str`
Compute SHA256 hash of file contents.

**Parameters**:
- `file_path`: Path to file

**Returns**: Hex string hash

**Purpose**: Detect file changes for cache invalidation

**Example**:
```python
hash1 = indexer.compute_file_hash("src/utils.py")
# Modify file
hash2 = indexer.compute_file_hash("src/utils.py")
if hash1 != hash2:
    print("File has changed, need to re-index")
```

#### Language-Specific Parsing

**Python (AST)**:
- Uses `ast.parse()` for accurate parsing
- Extracts function arguments, return types, decorators
- Analyzes function body for dependencies
- Gets docstrings from AST nodes

**JavaScript/TypeScript (Regex)**:
- Patterns: `function\s+(\w+)`, `(\w+)\s*:\s*function`, `(\w+)\s*=\s*\([^)]*\)\s*=>`
- Extracts function names and basic signatures
- Limited dependency extraction

**Other Languages (Regex)**:
- Language-specific function/class patterns
- Basic extraction of names and signatures
- No dependency analysis

#### Complexity Estimation

```python
def _estimate_complexity(code: str) -> str:
    lines = len(code.split("\n"))
    control_flow_count = count_keywords(["if", "for", "while", "switch"])
    nesting_depth = calculate_max_indentation()

    score = lines/10 + control_flow_count*2 + nesting_depth

    if score < 5: return "low"
    elif score < 15: return "medium"
    else: return "high"
```

---

### CodeMemoryStorage

**Location**: `memory_lib/codebase/code_storage.py`

**Purpose**: Specialized storage for code entities and documentation.

#### Responsibilities

- Store code entities with rich metadata
- Store non-code memories (documentation, debugging sessions)
- Provide code-specific queries (by file, language, entity)
- Maintain indices for performance
- Return unified results with type tagging

#### Database Schema

```sql
CREATE TABLE code_memories (
    id TEXT PRIMARY KEY,
    file_path TEXT,
    entity_name TEXT,
    code_snippet TEXT,
    docstring TEXT,
    signature TEXT,
    language TEXT,
    dependencies TEXT,      -- JSON array
    imports TEXT,           -- JSON array
    complexity TEXT,        -- "low" | "medium" | "high"
    last_modified TEXT,     -- ISO 8601
    timestamp TEXT NOT NULL,
    metadata TEXT           -- JSON object
);

CREATE TABLE non_code_memories (
    id TEXT PRIMARY KEY,
    category TEXT,          -- "documentation" | "debugging" | "general"
    title TEXT,
    content TEXT NOT NULL,
    file_path TEXT,
    timestamp TEXT NOT NULL,
    metadata TEXT           -- JSON object
);

-- Indices for performance
CREATE INDEX idx_code_file_path ON code_memories(file_path);
CREATE INDEX idx_code_entity_name ON code_memories(entity_name);
CREATE INDEX idx_code_language ON code_memories(language);
CREATE INDEX idx_code_last_modified ON code_memories(last_modified);
CREATE INDEX idx_non_code_category ON non_code_memories(category);
CREATE INDEX idx_non_code_file_path ON non_code_memories(file_path);
```

#### Public Methods

##### `__init__(self, db_path: str = "code_memories.db")`
Initialize code storage.

**Parameters**:
- `db_path`: Database file path

##### `add_code_memory(self, file_path, entity_name, code_snippet, **kwargs) -> str`
Add a code entity memory.

**Parameters**:
- `file_path`: Source file path (required)
- `entity_name`: Function/class name (required)
- `code_snippet`: Code text (required)
- `docstring`: Documentation string (optional)
- `signature`: Function signature (optional)
- `language`: Programming language (optional)
- `dependencies`: List of called functions (optional)
- `imports`: List of import statements (optional)
- `complexity`: "low"|"medium"|"high" (optional)
- `last_modified`: ISO timestamp (optional)
- `metadata`: Additional metadata dict (optional)

**Returns**: Memory ID

**Example**:
```python
storage = CodeMemoryStorage()
memory_id = storage.add_code_memory(
    file_path="src/utils.py",
    entity_name="calculate_sum",
    code_snippet="def calculate_sum(a, b):\n    return a + b",
    signature="calculate_sum(a, b)",
    language="python",
    dependencies=["validate_numbers"],
    complexity="low"
)
```

##### `add_non_code_memory(self, category: str, content: str, title: Optional[str] = None, file_path: Optional[str] = None, metadata: Optional[Dict] = None) -> str`
Add a non-code memory (documentation, debugging notes, etc.).

**Parameters**:
- `category`: "documentation" | "debugging" | "general"
- `content`: Memory content (required)
- `title`: Memory title (optional)
- `file_path`: Related file if any (optional)
- `metadata`: Additional metadata (optional)

**Returns**: Memory ID

**Example**:
```python
doc_id = storage.add_non_code_memory(
    category="documentation",
    title="API Authentication",
    content="Our API uses JWT tokens for authentication...",
    metadata={"section": "authentication", "version": "2.0"}
)
```

##### `get_all_memories(self) -> List[Dict]`
Get all memories (code + non-code) with type tags.

**Returns**: List of memory dicts, each with `"type": "code"` or `"type": "non-code"`

**Example**:
```python
all_memories = storage.get_all_memories()
code_count = sum(1 for m in all_memories if m["type"] == "code")
doc_count = sum(1 for m in all_memories if m["type"] == "non-code")
```

##### `get_memories_by_file(self, file_path: str) -> List[Dict]`
Get all memories related to a specific file.

**Parameters**:
- `file_path`: File path to query

**Returns**: List of memory dicts (code + non-code)

**Example**:
```python
memories = storage.get_memories_by_file("src/auth.py")
```

##### `get_memories_by_language(self, language: str) -> List[Dict]`
Get all code memories for a specific language.

**Parameters**:
- `language`: Language name (e.g., "python", "javascript")

**Returns**: List of code memory dicts

**Example**:
```python
python_memories = storage.get_memories_by_language("python")
```

##### `delete_memories_by_file(self, file_path: str) -> int`
Delete all memories related to a file.

**Parameters**:
- `file_path`: File path

**Returns**: Number of memories deleted

**Example**:
```python
deleted = storage.delete_memories_by_file("src/old_utils.py")
print(f"Deleted {deleted} memories")
```

---

### CodeMemoryRetrieval

**Location**: `memory_lib/codebase/code_retrieval.py`

**Purpose**: Enhanced retrieval with caching and code-specific optimizations.

#### Responsibilities

- Inherit base retrieval functionality
- Implement score caching with file-hash-based invalidation
- Apply dependency boosting
- Apply recency boosting
- Support CodeContext for rich queries
- Provide cache management

#### Data Model: CodeContext

```python
@dataclass
class CodeContext:
    user_query: str                      # Required: the actual question/task
    current_file: Optional[str] = None   # File user is working in
    recent_changes: Optional[str] = None # Recent edits
    errors: Optional[str] = None         # Error messages
    accessed_files: Optional[List[str]] = None  # Recently viewed files
    additional_context: Optional[str] = None    # Any other context
```

#### Public Methods

##### `__init__(self, small_model_fn, enable_caching=True, enable_dependency_boost=True, enable_recency_boost=True, dependency_boost_amount=0.15, recency_boost_amount=0.10, recency_days=7, **kwargs)`
Initialize code retrieval with optimizations.

**Parameters**:
- `small_model_fn`: Async LLM function
- `enable_caching`: Enable score caching (default True)
- `enable_dependency_boost`: Enable dependency boosting (default True)
- `enable_recency_boost`: Enable recency boosting (default True)
- `dependency_boost_amount`: Score boost for dependencies (default 0.15)
- `recency_boost_amount`: Score boost for recent files (default 0.10)
- `recency_days`: Days to consider "recent" (default 7)
- `**kwargs`: Additional parameters for base MemoryRetrieval

**Example**:
```python
retrieval = CodeMemoryRetrieval(
    small_model_fn=my_llm,
    enable_caching=True,
    dependency_boost_amount=0.20,
    recency_days=14
)
```

##### `async retrieve_code_memories(self, context: Union[str, CodeContext], memories: List[Dict], file_hashes: Optional[Dict[str, str]] = None) -> List[ScoredMemory]`
Main retrieval method with caching and boosting.

**Parameters**:
- `context`: String or CodeContext object
- `memories`: List of memory dicts from storage
- `file_hashes`: Dict mapping file_path → hash for cache keys

**Returns**: List of ScoredMemory objects

**Example**:
```python
from memory_lib.codebase import CodeContext

context = CodeContext(
    user_query="How does authentication work?",
    current_file="src/api/auth.py",
    errors="AttributeError: 'NoneType' object has no attribute 'id'"
)

memories = storage.get_all_memories()
file_hashes = {m.get("file_path"): indexer.compute_file_hash(m["file_path"])
               for m in memories if m.get("file_path")}

relevant = await retrieval.retrieve_code_memories(
    context, memories, file_hashes
)
```

##### `format_code_memories_for_prompt(self, memories: List[ScoredMemory], group_by_file: bool = True, include_scores: bool = True) -> str`
Format code memories with optional file grouping.

**Parameters**:
- `memories`: List of ScoredMemory objects
- `group_by_file`: Group by source file (default True)
- `include_scores`: Include relevance scores (default True)

**Returns**: Formatted string

**Example**:
```python
formatted = retrieval.format_code_memories_for_prompt(
    relevant,
    group_by_file=True
)
```

**Output format** (grouped):
```
=== src/auth.py ===

Function: validate_token (Relevance: 0.95)
Signature: validate_token(token: str) -> bool
Complexity: medium
Dependencies: decode_jwt, check_expiry

Code:
def validate_token(token: str) -> bool:
    ...

---

Function: decode_jwt (Relevance: 0.78)
...

=== src/utils.py ===
...
```

##### `invalidate_cache_for_file(self, file_path: str)`
Invalidate cached scores for a specific file.

**Parameters**:
- `file_path`: File whose cache entries should be cleared

**Example**:
```python
# File was modified
retrieval.invalidate_cache_for_file("src/auth.py")
```

##### `clear_cache(self)`
Clear entire score cache.

**Example**:
```python
retrieval.clear_cache()
```

##### `get_cache_stats(self) -> Dict`
Get cache statistics.

**Returns**: Dict with keys:
- `size`: Number of cached entries
- `hit_rate`: Cache hit percentage (if tracked)

**Example**:
```python
stats = retrieval.get_cache_stats()
print(f"Cache has {stats['size']} entries")
```

#### Optimization Details

**Caching**:
- Cache key: `(hash(context_string), memory_id, file_hash)`
- Cache value: `(score, reasoning, timestamp)`
- Max age: 3600 seconds (1 hour)
- Invalidation: Automatic on file hash change, manual via API

**Dependency Boosting**:
1. Identify memories with score ≥ threshold
2. Extract their dependencies
3. Boost scores of any memories matching those dependencies
4. Formula: `new_score = min(1.0, old_score + boost_amount)`

**Recency Boosting**:
1. Check last_modified timestamp
2. If within recency_days, boost score
3. Formula: `new_score = min(1.0, old_score + boost_amount)`

---

### CodeMemorySystem

**Location**: `memory_lib/codebase/code_memory_system.py`

**Purpose**: High-level facade for complete code intelligence.

#### Responsibilities

- Orchestrate indexing, storage, and retrieval
- Provide unified code API
- Handle re-indexing
- Manage cache lifecycle
- Support code-specific queries

#### Public Methods

##### `__init__(self, small_model_fn, db_path="code_memories.db", **retrieval_kwargs)`
Initialize code memory system.

**Parameters**:
- `small_model_fn`: Async LLM function
- `db_path`: Database path (default "code_memories.db")
- `**retrieval_kwargs`: Parameters for CodeMemoryRetrieval

**Example**:
```python
system = CodeMemorySystem(
    small_model_fn=my_llm,
    db_path="my_code.db",
    relevance_threshold=0.75,
    max_memories=20,
    enable_caching=True
)
```

##### `index_file(self, file_path: str) -> List[str]`
Index a file and store entities.

**Parameters**:
- `file_path`: Path to source file

**Returns**: List of created memory IDs

**Example**:
```python
memory_ids = system.index_file("src/utils.py")
print(f"Indexed {len(memory_ids)} entities")
```

##### `index_repository(self, directory: str, exclude_patterns: Optional[List[str]] = None, recursive: bool = True) -> List[str]`
Index entire repository.

**Parameters**:
- `directory`: Root directory
- `exclude_patterns`: Patterns to exclude
- `recursive`: Recurse into subdirectories

**Returns**: List of all created memory IDs

**Example**:
```python
memory_ids = system.index_repository(
    "src",
    exclude_patterns=["*/tests/*", "*/__pycache__/*", "*/.git/*"],
    recursive=True
)
print(f"Indexed repository: {len(memory_ids)} entities")
```

##### `reindex_file(self, file_path: str) -> List[str]`
Re-index a file (delete old entities, index anew).

**Parameters**:
- `file_path`: File to re-index

**Returns**: List of new memory IDs

**Example**:
```python
# File was modified
new_ids = system.reindex_file("src/auth.py")
```

##### `add_documentation_memory(self, title: str, content: str, category: str = "documentation", file_path: Optional[str] = None, metadata: Optional[Dict] = None) -> str`
Add documentation memory.

**Parameters**:
- `title`: Documentation title
- `content`: Documentation text
- `category`: Category (default "documentation")
- `file_path`: Related file (optional)
- `metadata`: Additional metadata (optional)

**Returns**: Memory ID

**Example**:
```python
doc_id = system.add_documentation_memory(
    title="Authentication Flow",
    content="Our system uses JWT tokens...",
    file_path="src/auth.py"
)
```

##### `add_debugging_session(self, description: str, error: str, solution: str, file_path: Optional[str] = None, metadata: Optional[Dict] = None) -> str`
Add debugging session memory.

**Parameters**:
- `description`: Problem description
- `error`: Error message
- `solution`: How it was solved
- `file_path`: Related file (optional)
- `metadata`: Additional metadata (optional)

**Returns**: Memory ID

**Example**:
```python
debug_id = system.add_debugging_session(
    description="AttributeError in validate_token",
    error="'NoneType' object has no attribute 'id'",
    solution="Added null check before accessing user.id",
    file_path="src/auth.py"
)
```

##### `async retrieve_relevant_memories(self, context: Union[str, CodeContext], include_file_hashes: bool = True) -> List[ScoredMemory]`
Retrieve relevant code memories.

**Parameters**:
- `context`: String or CodeContext
- `include_file_hashes`: Compute hashes for caching (default True)

**Returns**: List of ScoredMemory objects

**Example**:
```python
from memory_lib.codebase import CodeContext

context = CodeContext(
    user_query="Fix the authentication bug",
    current_file="src/auth.py",
    errors="NoneType has no attribute 'id'"
)

relevant = await system.retrieve_relevant_memories(context)
```

##### `async query(self, context: Union[str, CodeContext], primary_model_fn, custom_prompt_template: Optional[str] = None) -> str`
Complete query pipeline.

**Parameters**:
- `context`: Query context
- `primary_model_fn`: Async function for main LLM
- `custom_prompt_template`: Custom template (optional)

**Returns**: Primary LLM response

**Example**:
```python
response = await system.query(
    context=CodeContext(
        user_query="How do I add a new endpoint?",
        current_file="src/api/routes.py"
    ),
    primary_model_fn=my_gpt4_function
)
```

##### `clear_cache(self)`
Clear retrieval cache.

**Example**:
```python
system.clear_cache()
```

##### `get_stats(self) -> Dict`
Get system statistics.

**Returns**: Dict with:
- `total_code_memories`: Count of code entities
- `total_non_code_memories`: Count of documentation/debugging
- `total_memories`: Total count
- Cache stats, retrieval config

**Example**:
```python
stats = system.get_stats()
print(f"System has {stats['total_code_memories']} code entities")
print(f"Cache size: {stats['cache_stats']['size']}")
```

---

## Data Models

### Memory Dictionary (from Storage)

```python
{
    "id": str,              # UUID
    "text": str,            # Content
    "metadata": dict,       # Additional data
    "timestamp": str        # ISO 8601 UTC
}
```

### Code Memory Dictionary

```python
{
    "id": str,
    "type": "code",
    "file_path": str,
    "entity_name": str,
    "code_snippet": str,
    "docstring": str,
    "signature": str,
    "language": str,
    "dependencies": list[str],
    "imports": list[str],
    "complexity": str,       # "low" | "medium" | "high"
    "last_modified": str,    # ISO 8601
    "timestamp": str,
    "metadata": dict
}
```

### Non-Code Memory Dictionary

```python
{
    "id": str,
    "type": "non-code",
    "category": str,         # "documentation" | "debugging" | "general"
    "title": str,
    "content": str,
    "file_path": str,        # Optional
    "timestamp": str,
    "metadata": dict
}
```

### ScoredMemory

```python
@dataclass
class ScoredMemory:
    memory_id: str
    text: str
    metadata: Optional[Dict[str, Any]]
    timestamp: str
    relevance_score: float    # 0.0 to 1.0
    reasoning: str            # Why this score
```

### CodeContext

```python
@dataclass
class CodeContext:
    user_query: str                        # Required
    current_file: Optional[str] = None
    recent_changes: Optional[str] = None
    errors: Optional[str] = None
    accessed_files: Optional[List[str]] = None
    additional_context: Optional[str] = None
```

---

## Configuration

### MemoryRetrieval Configuration

```python
{
    "relevance_threshold": 0.7,    # 0.0 - 1.0
    "max_memories": 10,            # Positive integer
    "batch_size": 10,              # Concurrent API calls
    "retry_attempts": 3,           # API retry count
    "retry_delay": 1.0             # Seconds between retries
}
```

### CodeMemoryRetrieval Additional Configuration

```python
{
    "enable_caching": True,           # Enable score cache
    "enable_dependency_boost": True,  # Boost dependencies
    "enable_recency_boost": True,     # Boost recent files
    "dependency_boost_amount": 0.15,  # Score increase
    "recency_boost_amount": 0.10,     # Score increase
    "recency_days": 7                 # Days considered recent
}
```

### Usage Example

```python
# General system
system = MemorySystem(
    small_model_fn=my_llm,
    db_path="memories.db",
    relevance_threshold=0.8,
    max_memories=5,
    batch_size=15,
    retry_attempts=5
)

# Code system
code_system = CodeMemorySystem(
    small_model_fn=my_llm,
    db_path="code.db",
    relevance_threshold=0.75,
    max_memories=20,
    enable_caching=True,
    dependency_boost_amount=0.20,
    recency_days=14
)

# Runtime updates
system.update_retrieval_config(
    relevance_threshold=0.9,
    max_memories=3
)
```

---

**Next**: See [API Reference](API_REFERENCE.md) for complete public API documentation suitable for external release.
