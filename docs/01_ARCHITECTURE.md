# Memory Systems - Architecture Documentation

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Design Philosophy](#design-philosophy)
3. [Component Overview](#component-overview)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Design Patterns](#design-patterns)
7. [Performance Considerations](#performance-considerations)
8. [Scalability](#scalability)
9. [Security Considerations](#security-considerations)

## System Architecture

### High-Level Architecture

The memory system implements a **two-tier architecture** with distinct but related subsystems:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Memory Systems                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────┐  ┌─────────────────────────────┐  │
│  │   General Memory System    │  │   Code Memory System        │  │
│  ├────────────────────────────┤  ├─────────────────────────────┤  │
│  │                            │  │                             │  │
│  │  ┌──────────────────────┐ │  │  ┌───────────────────────┐ │  │
│  │  │  MemoryStorage       │ │  │  │  CodeIndexer          │ │  │
│  │  │  - SQLite            │ │  │  │  - AST parsing        │ │  │
│  │  │  - CRUD operations   │ │  │  │  - Multi-language     │ │  │
│  │  └──────────────────────┘ │  │  └───────────────────────┘ │  │
│  │                            │  │                             │  │
│  │  ┌──────────────────────┐ │  │  ┌───────────────────────┐ │  │
│  │  │  MemoryRetrieval     │ │  │  │  CodeMemoryStorage    │ │  │
│  │  │  - LLM scoring       │ │  │  │  - Dual tables        │ │  │
│  │  │  - Filtering         │ │  │  │  - Indices            │ │  │
│  │  │  - Top-K selection   │ │  │  └───────────────────────┘ │  │
│  │  └──────────────────────┘ │  │                             │  │
│  │             │              │  │  ┌───────────────────────┐ │  │
│  │             ▼              │  │  │  CodeMemoryRetrieval  │ │  │
│  │  ┌──────────────────────┐ │  │  │  - Caching            │ │  │
│  │  │    MemorySystem      │ │  │  │  - Boost mechanisms   │ │  │
│  │  │    (Facade)          │ │  │  └───────────────────────┘ │  │
│  │  └──────────────────────┘ │  │             │               │  │
│  │                            │  │             ▼               │  │
│  └────────────────────────────┘  │  ┌───────────────────────┐ │  │
│                                   │  │  CodeMemorySystem     │ │  │
│                                   │  │  (Facade)             │ │  │
│                                   │  └───────────────────────┘ │  │
│                                   └─────────────────────────────┘  │
│                                                                       │
│                         Common Infrastructure                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│  │  LLM Interface  │  │  SQLite Storage  │  │  Config Manager │   │
│  │  (abstracted)   │  │  (persistence)   │  │  (parameters)   │   │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Layered Architecture

The system follows a **three-layer architecture**:

#### Layer 1: Data Layer
- **Responsibility**: Persistence and data management
- **Components**: SQLite databases, file system for code indexing
- **Characteristics**:
  - ACID compliance through SQLite transactions
  - Indexed queries for performance
  - Schema versioning for migrations

#### Layer 2: Logic Layer
- **Responsibility**: Business logic and algorithms
- **Components**: Retrieval algorithms, scoring logic, indexing logic
- **Characteristics**:
  - Stateless operations (except caching)
  - Async/await for I/O operations
  - Error handling and retry logic

#### Layer 3: API Layer
- **Responsibility**: User-facing interfaces
- **Components**: MemorySystem and CodeMemorySystem facades
- **Characteristics**:
  - Simple, intuitive methods
  - Sensible defaults
  - Configuration flexibility

## Design Philosophy

### Core Principles

1. **Composition Over Inheritance**
   - Systems compose smaller, focused components
   - Enables flexible recombination
   - Reduces coupling

2. **Dependency Injection**
   - LLM functions injected at runtime
   - Easy testing with mocks
   - Provider agnostic

3. **Single Responsibility**
   - Each component has one clear purpose
   - Storage handles persistence
   - Retrieval handles scoring
   - Indexer handles code analysis

4. **Fail-Safe Defaults**
   - Sensible default parameters
   - Graceful degradation
   - Comprehensive error messages

5. **Performance by Design**
   - Async operations throughout
   - Batch processing for efficiency
   - Strategic caching where beneficial

### Key Design Decisions

#### Decision 1: Exhaustive Scoring vs. Approximate Search

**Choice**: Exhaustive scoring with small LLM

**Alternatives considered**:
- Vector embeddings + cosine similarity
- BM25 text search
- Hybrid approaches

**Rationale**:
- **Higher precision**: LLM understands semantic relevance better than embeddings
- **Explainability**: Each memory has a score and reasoning
- **Flexibility**: Works with any type of memory content
- **Cost-effective**: Small models (GPT-3.5, Haiku) are inexpensive
- **Scalability**: Batching keeps latency acceptable

**Trade-offs**:
- Higher latency than embedding search (mitigated by batching and caching)
- Higher API costs (but still cost-effective overall)

#### Decision 2: SQLite vs. Vector Database

**Choice**: SQLite for storage

**Alternatives considered**:
- PostgreSQL with pgvector
- Pinecone / Weaviate
- Redis with RediSearch

**Rationale**:
- **Simplicity**: No external dependencies
- **Portability**: Database is a single file
- **Performance**: Excellent for <1M records
- **ACID compliance**: Built-in transactions
- **Cost**: Zero infrastructure cost

**Trade-offs**:
- Not suitable for >1M memories (acceptable for target use cases)
- No distributed queries (not needed for single-application use)

#### Decision 3: Code Indexing Approach

**Choice**: AST parsing for Python, regex for others

**Alternatives considered**:
- Tree-sitter for all languages
- Language servers (LSP)
- Static analysis tools

**Rationale**:
- **Accuracy**: AST provides perfect parsing for Python
- **Simplicity**: Regex adequate for basic extraction in other languages
- **No dependencies**: No external parsers required
- **Extensibility**: Easy to add Tree-sitter later if needed

**Trade-offs**:
- Less accurate for non-Python languages (acceptable for v1)
- Doesn't understand complex language features (acceptable for function/class extraction)

## Component Overview

### General Memory System Components

#### MemoryStorage
**Purpose**: Persistent storage for memories

**Responsibilities**:
- Store memories with metadata and timestamps
- Provide CRUD operations
- Maintain data integrity
- Support queries and filtering

**Key characteristics**:
- Single table schema
- UUID primary keys
- JSON metadata field for flexibility
- ISO 8601 timestamps
- Thread-safe operations

#### MemoryRetrieval
**Purpose**: Intelligent memory retrieval using LLM scoring

**Responsibilities**:
- Score all memories for relevance
- Filter by threshold
- Select top-K memories
- Format memories for prompts

**Key characteristics**:
- Async batch processing
- Configurable thresholds and limits
- Retry logic for API failures
- Structured output parsing

#### MemorySystem
**Purpose**: Unified facade for memory operations

**Responsibilities**:
- Orchestrate storage and retrieval
- Provide high-level API
- Manage configuration
- Handle the complete query pipeline

**Key characteristics**:
- Composition of Storage + Retrieval
- Context manager support
- Configuration updates at runtime
- Statistics and monitoring

### Code Memory System Components

#### CodeIndexer
**Purpose**: Extract code entities from source files

**Responsibilities**:
- Parse source files
- Extract functions, classes, imports
- Compute file hashes for change detection
- Support multiple languages

**Key characteristics**:
- Language-specific strategies
- Dependency extraction
- Complexity estimation
- Recursive directory traversal

#### CodeMemoryStorage
**Purpose**: Specialized storage for code and documentation

**Responsibilities**:
- Store code entities with rich metadata
- Store non-code memories (docs, debugging sessions)
- Provide code-specific queries
- Maintain indices for performance

**Key characteristics**:
- Two-table schema (code + non-code)
- Multiple indices (file, entity, language)
- JSON fields for structured data
- Type tagging for unified queries

#### CodeMemoryRetrieval
**Purpose**: Enhanced retrieval with code-specific optimizations

**Responsibilities**:
- Score memories (inherits from base)
- Cache scores for performance
- Apply dependency boosting
- Apply recency boosting

**Key characteristics**:
- Extends base retrieval
- In-memory score cache
- Cache invalidation on file changes
- Context-aware boosting

#### CodeMemorySystem
**Purpose**: Unified facade for code intelligence

**Responsibilities**:
- Orchestrate indexing, storage, retrieval
- Provide high-level code API
- Manage cache lifecycle
- Handle code-specific queries

**Key characteristics**:
- Composition of Indexer + Storage + Retrieval
- CodeContext support
- Re-indexing capabilities
- Cache management

## Data Flow

### General Memory Query Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Query                                │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  MemorySystem.query()         │
                │  - Receive context & task     │
                └───────────────┬───────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
┌───────────────────────────┐   ┌──────────────────────────────┐
│  Storage.get_all_memories │   │  Context formatting          │
│  - Retrieve all memories  │   │  - Prepare retrieval prompt  │
└───────────┬───────────────┘   └──────────────┬───────────────┘
            │                                   │
            └────────────┬──────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  Retrieval.retrieve_relevant()     │
        │  - Score all memories in batches   │
        └────────────┬───────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐      ┌─────────────────────┐
│ Batch 1       │ ...  │ Batch N             │
│ - Score M1    │      │ - Score MN          │
│ - Score M2    │      │ - Score MN+1        │
│ - ...         │      │ - ...               │
└───────┬───────┘      └─────────┬───────────┘
        │                        │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Filter & Select           │
        │  - Keep score >= threshold │
        │  - Sort by score           │
        │  - Take top K              │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Format for prompt         │
        │  - Create context string   │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Primary LLM call          │
        │  - Send context + task     │
        │  - Receive response        │
        └────────────┬───────────────┘
                     │
                     ▼
                ┌─────────┐
                │ Response│
                └─────────┘
```

### Code Memory Query Flow with Optimizations

```
┌──────────────────────────────────────────────────────────────────┐
│               CodeMemorySystem.query(CodeContext)                │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  Build rich context           │
                │  - Query + file + errors +... │
                └───────────────┬───────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────┐
        │  CodeStorage.get_all_memories()        │
        │  - Retrieve code + non-code memories   │
        │  - Include file hashes                 │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  CodeRetrieval.retrieve_code_memories()│
        └────────────────┬───────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
  ┌──────────────────┐      ┌──────────────────┐
  │ Check cache      │      │ Score uncached   │
  │ - Compute hashes │      │ - Batch process  │
  │ - Return cached  │      │ - Store in cache │
  └────────┬─────────┘      └─────────┬────────┘
           │                          │
           └────────────┬─────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Apply dependency boosting    │
        │  - Find relevant dependencies │
        │  - Boost related entities     │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Apply recency boosting       │
        │  - Check last_modified        │
        │  - Boost recent files         │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Filter & Select              │
        │  - Threshold filter           │
        │  - Top-K selection            │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Format for prompt            │
        │  - Group by file (optional)   │
        │  - Include metadata           │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Primary LLM call             │
        │  - Rich code context          │
        └───────────────┬───────────────┘
                        │
                        ▼
                    ┌────────┐
                    │Response│
                    └────────┘
```

## Technology Stack

### Core Technologies

- **Python 3.9+**: Modern async/await support, type hints
- **SQLite 3**: Embedded database, ACID compliance
- **asyncio**: Asynchronous I/O for LLM API calls

### Standard Library Components

- `ast`: Python Abstract Syntax Tree parsing
- `sqlite3`: Database interface
- `hashlib`: File hashing (SHA256)
- `pathlib`: Path manipulation
- `json`: Metadata serialization
- `datetime`: Timestamp management
- `re`: Regular expressions for parsing
- `uuid`: Unique ID generation

### External Dependencies

None required for core functionality. LLM providers are user-supplied:
- OpenAI SDK (optional, user choice)
- Anthropic SDK (optional, user choice)
- Any async LLM function

### Development Dependencies

- `pytest`: Testing framework
- `pytest-asyncio`: Async test support
- `black`: Code formatting
- `mypy`: Type checking
- `pylint`: Linting

## Design Patterns

### 1. Facade Pattern

**Where**: `MemorySystem`, `CodeMemorySystem`

**Purpose**: Provide simple interfaces to complex subsystems

**Benefits**:
- Hide implementation complexity
- Provide sensible defaults
- Easy to use for common cases
- Flexible for advanced cases

### 2. Strategy Pattern

**Where**: LLM function injection

**Purpose**: Make LLM provider interchangeable

**Implementation**:
```python
async def score_memory(small_model_fn, context, memory):
    response = await small_model_fn(prompt)
    return parse_score(response)
```

**Benefits**:
- Works with any LLM provider
- Easy to test with mocks
- No vendor lock-in

### 3. Template Method Pattern

**Where**: `MemoryRetrieval` → `CodeMemoryRetrieval`

**Purpose**: Define algorithm skeleton, allow extensions

**Implementation**:
```python
class MemoryRetrieval:
    async def retrieve_relevant_memories(self, context, memories):
        scored = await self.score_all_memories(context, memories)
        return self.filter_and_select(scored)

class CodeMemoryRetrieval(MemoryRetrieval):
    async def retrieve_code_memories(self, context, memories, file_hashes):
        # Check cache first
        scored = await super().retrieve_relevant_memories(...)
        # Apply boosting
        scored = self.apply_dependency_boost(scored)
        scored = self.apply_recency_boost(scored)
        return scored
```

**Benefits**:
- Code reuse
- Consistent base behavior
- Extensibility

### 4. Dependency Injection

**Where**: Throughout the system

**Purpose**: Loose coupling, testability

**Examples**:
- LLM functions injected into systems
- Storage injected into systems
- Retrieval injected into systems

**Benefits**:
- Easy unit testing
- Flexible configuration
- Runtime behavior changes

### 5. Repository Pattern

**Where**: `MemoryStorage`, `CodeMemoryStorage`

**Purpose**: Abstract data access

**Benefits**:
- Encapsulate database logic
- Easy to swap storage backends
- Clean separation of concerns

## Performance Considerations

### Latency Optimization

1. **Batch Processing**
   - Score multiple memories concurrently
   - Default batch size: 10
   - Configurable based on rate limits

2. **Caching** (Code System)
   - In-memory score cache
   - Cache key: (context, memory, file_hash)
   - Automatic invalidation on file changes
   - Typical cache hit rate: 60-80% after warm-up

3. **Async Operations**
   - Non-blocking I/O throughout
   - Concurrent LLM API calls
   - Efficient resource utilization

### Cost Optimization

1. **Small Model for Scoring**
   - GPT-3.5-turbo: ~$0.0005 per memory
   - Claude Haiku: ~$0.00025 per memory
   - Total: $0.05-0.08 per query (100 memories)

2. **Caching** (Code System)
   - Reduces API calls by 60-80%
   - Pay once, reuse scores
   - Especially valuable for common queries

3. **Efficient Prompts**
   - Minimal token usage for scoring
   - Structured output for reliable parsing
   - No unnecessary context

### Storage Optimization

1. **SQLite Efficiency**
   - Indexed columns for fast queries
   - JSON metadata for flexibility without joins
   - Vacuuming for space reclamation

2. **Memory Footprint**
   - Code memory: ~5-10 KB per function
   - General memory: ~1-2 KB per item
   - Cache: ~500 bytes per cached score

## Scalability

### Vertical Scalability

**Supported scale**:
- 10,000-100,000 memories per database
- 100-1,000 concurrent queries per second (with caching)
- Limited by SQLite's single-writer constraint

**Bottlenecks**:
- LLM API rate limits
- SQLite write throughput
- Memory for caching

**Mitigations**:
- Batch size tuning
- Read replicas (SQLite supports multiple readers)
- Cache size limits

### Horizontal Scalability

**Not currently supported**:
- Distributed storage
- Sharding
- Multi-region deployment

**Future considerations**:
- Replace SQLite with PostgreSQL for writes
- Add Redis for distributed caching
- Implement memory sharding by category/domain

## Security Considerations

### Data Security

1. **SQL Injection Prevention**
   - Parameterized queries throughout
   - No string concatenation for SQL
   - Input validation on IDs

2. **File System Access**
   - Path validation in CodeIndexer
   - No arbitrary file access
   - Exclude patterns for sensitive files

3. **Metadata Sanitization**
   - JSON encoding prevents injection
   - Validation of user-provided metadata
   - Size limits on stored data

### API Security

1. **LLM API Keys**
   - User-managed credentials
   - No storage of API keys in library
   - Secure transmission (HTTPS by provider)

2. **Rate Limiting**
   - Configurable batch sizes
   - Retry with exponential backoff
   - Respect provider rate limits

### Privacy Considerations

1. **Local Storage**
   - All data stored locally by default
   - No external transmission except to user-configured LLM APIs
   - Easy to encrypt database file

2. **Code Indexing**
   - Exclude patterns for sensitive files (.env, credentials, etc.)
   - Hash-based change detection (no code sent unless needed)
   - User control over what gets indexed

## Architecture Evolution

### Current State (v1)

- Single-machine deployment
- SQLite storage
- Synchronous indexing
- In-process caching

### Near-term Roadmap (v2)

- Async indexing with progress tracking
- Incremental re-indexing (detect changes automatically)
- Persistent cache (SQLite or Redis)
- Multi-language support improvements (Tree-sitter)

### Long-term Vision (v3+)

- Distributed deployment support
- PostgreSQL backend option
- Advanced code analysis (call graphs, type inference)
- Real-time indexing with file watchers
- Multi-modal support (images, diagrams in documentation)

---

**Next**: See [Component Documentation](02_COMPONENTS.md) for detailed component specifications.
