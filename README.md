# Memory System

A comprehensive memory solution with two variants:

1. **General Memory System**: Base library for storing and retrieving memories using LLM-based relevance scoring
2. **Codebase-Specific Memory System**: Specialized version for code intelligence with indexing, caching, and optimization

Both systems use an **exhaustive reasoning approach** where a small LLM scores the relevance of all memories, enabling precise selection of the most relevant context.

## Features

### General Memory System

- **Persistent Storage**: SQLite-based storage for memories with metadata
- **LLM-Based Retrieval**: Uses small model to score relevance of every memory
- **Filtering & Selection**: Threshold-based filtering and top-K selection
- **Async/Parallel**: Batched parallel API calls for efficiency
- **Easy API**: Simple, well-documented API for adding, retrieving, and querying memories

### Codebase-Specific Memory System

All features from the general system, plus:

- **Code Indexing**: Automatic extraction of functions, classes, and documentation from codebases
- **Multi-Language Support**: Python (AST-based), JavaScript/TypeScript, and more
- **Smart Caching**: Cache relevance scores for unchanged code
- **Dependency Awareness**: Automatically boost scores for related code entities
- **Recency Weighting**: Boost recently modified files
- **Code Context**: Rich context including current file, errors, recent changes

## Installation

```bash
# Clone or download the repository
cd memory

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### General Memory System

```python
import asyncio
from memory_lib import MemorySystem

async def small_model_api(prompt: str) -> str:
    """Call your small LLM (e.g., GPT-3.5, Claude Haiku)"""
    # Your API call here
    return response

async def primary_model_api(prompt: str) -> str:
    """Call your primary LLM (e.g., GPT-4, Claude Opus)"""
    # Your API call here
    return response

async def main():
    # Initialize
    memory = MemorySystem(
        small_model_fn=small_model_api,
        db_path="my_memories.db",
        relevance_threshold=0.7,
        max_memories=10
    )

    # Add memories
    memory.add_memory(
        "Python uses dynamic typing",
        metadata={"topic": "python", "category": "basics"}
    )

    # Query
    response = await memory.query(
        context="Tell me about Python",
        task="Explain Python's key features",
        primary_model_fn=primary_model_api
    )
    print(response)

asyncio.run(main())
```

### Codebase-Specific Memory System

```python
import asyncio
from memory_lib import CodeMemorySystem
from memory_lib.codebase import CodeContext

async def small_model_api(prompt: str) -> str:
    """Call your small LLM for scoring"""
    return response

async def primary_model_api(prompt: str) -> str:
    """Call your primary LLM for main tasks"""
    return response

async def main():
    # Initialize
    code_memory = CodeMemorySystem(
        small_model_fn=small_model_api,
        db_path="code_memories.db",
        enable_caching=True,
        enable_dependency_boost=True
    )

    # Index codebase
    code_memory.index_repository(
        "src/",
        exclude_patterns=["*/tests/*", "*/__pycache__/*"]
    )

    # Create context
    context = CodeContext(
        user_query="Fix the authentication bug",
        current_file="api/auth.py",
        errors="AttributeError in validate_token"
    )

    # Query with context
    response = await code_memory.query(
        context=context,
        primary_model_fn=primary_model_api
    )
    print(response)

asyncio.run(main())
```

## Architecture

### General Memory System

```
┌─────────────────────────────────────────────────────────────┐
│                       MemorySystem                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐           ┌──────────────────┐      │
│  │  MemoryStorage   │           │ MemoryRetrieval  │      │
│  ├──────────────────┤           ├──────────────────┤      │
│  │ - SQLite DB      │           │ - Score all      │      │
│  │ - CRUD ops       │           │   memories       │      │
│  │ - Metadata       │           │ - Filter         │      │
│  └──────────────────┘           │ - Top-K select   │      │
│                                 └──────────────────┘      │
│                                          │                  │
│                                          ▼                  │
│                              ┌──────────────────┐          │
│                              │   Small Model    │          │
│                              │  (GPT-3.5, etc)  │          │
│                              └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Codebase-Specific Memory System

```
┌──────────────────────────────────────────────────────────────┐
│                    CodeMemorySystem                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐  ┌──────────────────┐  ┌─────────────┐│
│  │  CodeIndexer   │  │ CodeMemoryStorage│  │   CodeMemory││
│  ├────────────────┤  ├──────────────────┤  │  Retrieval  ││
│  │ - Parse code   │  │ - Code entities  │  │ - Caching   ││
│  │ - Extract      │  │ - Non-code       │  │ - Dependency││
│  │   functions    │  │   memories       │  │   boost     ││
│  │ - Multi-lang   │  │ - Specialized    │  │ - Recency   ││
│  └────────────────┘  │   metadata       │  │   boost     ││
│                      └──────────────────┘  └─────────────┘│
└──────────────────────────────────────────────────────────────┘
```

## Retrieval Pipeline

The memory system uses an exhaustive reasoning approach:

1. **Score All Memories**: Small model scores relevance of every memory (0-1)
2. **Filter**: Keep only memories with score ≥ threshold (default: 0.7)
3. **Select Top-K**: If more than K pass filter, take top K by score
4. **Format**: Format selected memories for inclusion in prompt
5. **Query Primary Model**: Call primary model with memories + task

### Cost Analysis

Per query:
- **N** calls to small model (one per memory)
- **1** call to primary model

For N=100 memories:
- Small model: 100 × $0.0005 = $0.05
- Primary model: 1 × $0.03 = $0.03
- **Total**: ~$0.08 per query

Optimizations reduce cost:
- **Caching** (code system): Skip scoring unchanged files
- **Batching**: Parallel API calls reduce latency

## API Documentation

### MemorySystem (General)

#### Initialization

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

#### Methods

- `add_memory(text, metadata, memory_id)` - Add a memory
- `get_memory(memory_id)` - Get specific memory
- `get_all_memories()` - Get all memories
- `update_memory(memory_id, text, metadata)` - Update memory
- `delete_memory(memory_id)` - Delete memory
- `retrieve_relevant_memories(context)` - Retrieve relevant memories
- `query(context, task, primary_model_fn)` - Complete query pipeline
- `update_retrieval_config(...)` - Update configuration
- `get_stats()` - Get system statistics

### CodeMemorySystem (Codebase-Specific)

#### Initialization

```python
CodeMemorySystem(
    small_model_fn: Callable[[str], Awaitable[str]],
    db_path: str = "code_memories.db",
    relevance_threshold: float = 0.7,
    max_memories: int = 15,
    batch_size: int = 10,
    enable_caching: bool = True,
    enable_dependency_boost: bool = True,
    enable_recency_boost: bool = True
)
```

#### Indexing Methods

- `index_file(file_path, auto_store)` - Index single file
- `index_repository(directory, exclude_patterns, recursive)` - Index entire repo
- `reindex_file(file_path)` - Re-index modified file

#### Storage Methods

- `add_code_memory(**kwargs)` - Add code memory manually
- `add_documentation_memory(title, content, category)` - Add documentation
- `add_debugging_session(title, content, metadata)` - Add debugging session
- `get_memory(memory_id)` - Get specific memory
- `get_all_memories()` - Get all memories
- `get_memories_by_file(file_path)` - Get memories for file
- `delete_memory(memory_id)` - Delete memory

#### Retrieval Methods

- `retrieve_relevant_memories(context, file_hashes)` - Retrieve with code context
- `query(context, primary_model_fn)` - Complete query pipeline
- `query_with_custom_prompt(...)` - Query with custom template

#### Configuration Methods

- `update_retrieval_config(...)` - Update retrieval settings
- `get_stats()` - Get system statistics
- `clear_cache()` - Clear score cache
- `clear_all_memories()` - Delete all memories

### CodeContext

```python
CodeContext(
    user_query: str,
    current_file: Optional[str] = None,
    recent_changes: Optional[str] = None,
    errors: Optional[str] = None,
    accessed_files: List[str] = [],
    additional_context: Optional[str] = None
)
```

## Examples

See the `examples/` directory:

- `general_memory_example.py` - Complete example for general memory system
- `code_memory_example.py` - Complete example for code memory system

Run examples:

```bash
cd examples
python general_memory_example.py
python code_memory_example.py
```

## Testing

```bash
cd tests
python -m pytest test_general.py
python -m pytest test_codebase.py
```

## Configuration

### Relevance Threshold

Controls minimum score for inclusion (0-1). Higher = more selective.

- **0.6**: Inclusive, many memories
- **0.7**: Balanced (default)
- **0.8**: Selective, high-quality only

### Max Memories (K)

Maximum memories to include in prompt.

- General system: 10 (default)
- Code system: 15 (default, code needs more context)

### Caching (Code System)

Cache relevance scores for unchanged files.

- Enabled by default
- Cache max age: 1 hour
- Invalidate on file modification

### Dependency Boost (Code System)

Boost scores for entities called by relevant functions.

- Enabled by default
- Boost factor: 0.15

### Recency Boost (Code System)

Boost scores for recently modified files.

- Enabled by default
- Boost factor: 0.1
- Recent threshold: 7 days

## Supported Languages (Code System)

- **Python**: Full AST-based parsing
- **JavaScript/TypeScript**: Regex-based extraction
- **Java, C/C++, Go, Rust, Ruby, PHP, C#**: Basic support

Easily extendable for additional languages.

## Performance

### Storage

- SQLite with indices for fast queries
- Handles 10,000+ memories efficiently

### Retrieval

- Parallel API calls (configurable batch size)
- Typical latency for 100 memories: 2-5 seconds
- Caching reduces repeat queries to milliseconds

## Best Practices

1. **Small Model Selection**: Use fast, cheap models (GPT-3.5-turbo, Claude Haiku)
2. **Batch Size**: Tune based on API rate limits (default: 10)
3. **Caching**: Enable for code systems to reduce costs
4. **Exclusion Patterns**: Exclude test files, generated code, dependencies
5. **Threshold Tuning**: Start at 0.7, adjust based on results
6. **Memory Granularity**: Index functions/classes, not entire files

## Troubleshooting

### High API costs

- Enable caching (code system)
- Reduce batch size
- Increase relevance threshold

### Missing relevant memories

- Lower relevance threshold
- Increase max_memories
- Check if code is indexed

### Slow retrieval

- Increase batch_size for more parallelism
- Reduce number of memories
- Check network/API latency

## Future Enhancements

- Vector embeddings for hybrid retrieval
- Semantic code search
- Cross-file dependency tracking
- Integration with LSP servers
- Web UI for memory management

## License

MIT

## Contributing

Contributions welcome! Please submit issues and pull requests.

## Authors

Developed as part of the Memory System project.
