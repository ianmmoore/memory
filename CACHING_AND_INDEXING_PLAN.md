# Memory Caching and Smart Indexing Plan

## Executive Summary

Implement advanced caching and smart indexing to reduce API costs by 70-80% while maintaining or improving memory retrieval quality. The strategy combines multi-level caching, intelligent pre-filtering, and task-aware indexing.

## Current State Analysis

### What We Have ✅

**Basic Caching** (`memory_lib/codebase/code_retrieval.py`):
```python
class ScoreCache:
    """Cache for relevance scores."""
    cache: Dict[tuple, tuple[float, str, float]]  # (context_hash, memory_id, file_hash) -> (score, reasoning, timestamp)
```

- Caches scores for code memories
- Invalidates on file changes
- Time-based expiration (1 hour default)

### Limitations ❌

1. **Cache key too specific**: `context_hash` includes full query, so similar queries don't benefit
2. **No pre-filtering**: Still scores ALL memories, just faster when cached
3. **No task categorization**: Can't filter memories by task type
4. **No semantic indexing**: Can't do similarity-based pre-filtering
5. **Memory growth unchecked**: As memories grow, even cached scoring gets expensive

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced Memory System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Layer 1: Smart Pre-Filtering                │  │
│  │  - Task categorization                                   │  │
│  │  - Keyword matching                                      │  │
│  │  - Metadata filtering                                    │  │
│  │  Result: 1000 memories → 200 relevant candidates        │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────────┐  │
│  │              Layer 2: Embedding-Based Search             │  │
│  │  - Semantic similarity (optional, if embeddings exist)   │  │
│  │  - Fast vector search                                    │  │
│  │  Result: 200 candidates → 50 highly relevant            │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────────┐  │
│  │         Layer 3: Multi-Level Score Caching               │  │
│  │  - L1: Exact cache (context + memory)                   │  │
│  │  - L2: Similar context cache                            │  │
│  │  - L3: Static memory features cache                     │  │
│  │  Result: 50 candidates, 40 from cache, 10 need scoring  │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────────┐  │
│  │            Layer 4: LLM Scoring (Only 10 calls)          │  │
│  │  - Score remaining un-cached memories                   │  │
│  │  - Store results in all cache levels                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Smart Indexing (Week 1)

### Goal: Pre-filter memories before scoring

### 1.1 Task Categorization System

**Implementation**: `memory_lib/indexing/task_categorizer.py`

```python
class TaskCategory(Enum):
    """Terminal-Bench task categories."""
    NETWORK = "network"
    DATA_ANALYSIS = "data_analysis"
    FILE_OPERATIONS = "file_operations"
    PYTHON_SCRIPTING = "python_scripting"
    API_INTEGRATION = "api_integration"
    SECURITY = "security"
    SYSTEM_ADMIN = "system_admin"
    UNKNOWN = "unknown"

class TaskCategorizer:
    """Categorizes tasks based on keywords and patterns."""

    CATEGORY_KEYWORDS = {
        TaskCategory.NETWORK: [
            "network", "ip", "ping", "curl", "http", "api endpoint",
            "port", "socket", "connection", "firewall"
        ],
        TaskCategory.DATA_ANALYSIS: [
            "csv", "json", "data", "parse", "analyze", "chart",
            "statistics", "plot", "dataframe", "pandas"
        ],
        # ... more categories
    }

    def categorize(self, task_description: str) -> TaskCategory:
        """Categorize task based on description."""
        # Score each category
        scores = {}
        desc_lower = task_description.lower()

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            scores[category] = score

        # Return highest scoring category
        best_category = max(scores.items(), key=lambda x: x[1])
        return best_category[0] if best_category[1] > 0 else TaskCategory.UNKNOWN
```

**Memory Schema Update**: `memory_lib/codebase/code_storage.py`

```python
# Add category field to memories
def add_code_memory(self, ..., categories: List[str] = None):
    """Add code memory with categories."""
    # Store categories for filtering
    metadata["categories"] = categories or []

def add_documentation_memory(self, ..., categories: List[str] = None):
    """Add documentation with categories."""
    metadata["categories"] = categories or []
```

**Index Creation**:
```sql
-- Add category index for fast filtering
CREATE INDEX idx_categories ON code_memories((metadata->>'categories'));
CREATE INDEX idx_categories_noncode ON non_code_memories((metadata->>'categories'));
```

### 1.2 Keyword-Based Pre-Filter

**Implementation**: `memory_lib/indexing/keyword_filter.py`

```python
class KeywordFilter:
    """Pre-filter memories using keyword matching."""

    def __init__(self):
        self.stop_words = {"the", "a", "an", "is", "are", "was", "were"}

    def extract_keywords(self, text: str) -> Set[str]:
        """Extract important keywords from text."""
        # Remove stop words, extract meaningful terms
        words = text.lower().split()
        keywords = {w for w in words if w not in self.stop_words and len(w) > 3}
        return keywords

    def score_keyword_overlap(
        self,
        query_keywords: Set[str],
        memory_text: str
    ) -> float:
        """Fast keyword-based relevance score."""
        memory_keywords = self.extract_keywords(memory_text)

        if not query_keywords or not memory_keywords:
            return 0.0

        overlap = query_keywords & memory_keywords
        return len(overlap) / len(query_keywords)

    def filter_memories(
        self,
        query: str,
        memories: List[Dict],
        min_overlap: float = 0.1,
        max_results: int = 100
    ) -> List[Dict]:
        """Pre-filter memories by keyword overlap."""
        query_keywords = self.extract_keywords(query)

        # Score all memories (fast, no API calls)
        scored = []
        for memory in memories:
            text = memory.get("text", "") + " " + str(memory.get("metadata", {}))
            score = self.score_keyword_overlap(query_keywords, text)
            if score >= min_overlap:
                scored.append((score, memory))

        # Sort by score and take top results
        scored.sort(reverse=True)
        return [mem for _, mem in scored[:max_results]]
```

### 1.3 Metadata-Based Filtering

**Implementation**: `memory_lib/indexing/metadata_filter.py`

```python
class MetadataFilter:
    """Filter memories by metadata attributes."""

    def filter_by_language(
        self,
        memories: List[Dict],
        languages: List[str]
    ) -> List[Dict]:
        """Filter to specific programming languages."""
        return [
            m for m in memories
            if m.get("language") in languages
        ]

    def filter_by_recency(
        self,
        memories: List[Dict],
        days: int = 30
    ) -> List[Dict]:
        """Filter to recently created/modified memories."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()

        return [
            m for m in memories
            if m.get("timestamp", "") > cutoff_iso or
               m.get("last_modified", "") > cutoff_iso
        ]

    def filter_by_success(
        self,
        memories: List[Dict]
    ) -> List[Dict]:
        """Filter to only successful solutions."""
        return [
            m for m in memories
            if m.get("metadata", {}).get("success", False)
        ]

    def apply_filters(
        self,
        memories: List[Dict],
        filters: Dict[str, Any]
    ) -> List[Dict]:
        """Apply multiple filters."""
        result = memories

        if "categories" in filters:
            result = self._filter_by_categories(result, filters["categories"])

        if "languages" in filters:
            result = self.filter_by_language(result, filters["languages"])

        if "recency_days" in filters:
            result = self.filter_by_recency(result, filters["recency_days"])

        if "success_only" in filters and filters["success_only"]:
            result = self.filter_by_success(result)

        return result
```

### 1.4 Combined Smart Index

**Implementation**: `memory_lib/indexing/smart_index.py`

```python
class SmartMemoryIndex:
    """Combined indexing system for intelligent pre-filtering."""

    def __init__(self):
        self.categorizer = TaskCategorizer()
        self.keyword_filter = KeywordFilter()
        self.metadata_filter = MetadataFilter()

    def pre_filter(
        self,
        query: str,
        memories: List[Dict],
        target_count: int = 50
    ) -> List[Dict]:
        """Intelligently pre-filter memories before LLM scoring.

        This reduces 1000 memories to ~50 high-quality candidates
        using fast, non-LLM methods.
        """
        # Step 1: Categorize query
        category = self.categorizer.categorize(query)

        # Step 2: Filter by category (if detected)
        if category != TaskCategory.UNKNOWN:
            category_filtered = [
                m for m in memories
                if category.value in m.get("metadata", {}).get("categories", [])
            ]
            # Fall back to all memories if category filter too aggressive
            if len(category_filtered) > 10:
                memories = category_filtered

        # Step 3: Filter by metadata
        metadata_filters = self._infer_metadata_filters(query)
        memories = self.metadata_filter.apply_filters(memories, metadata_filters)

        # Step 4: Keyword-based filtering
        memories = self.keyword_filter.filter_memories(
            query,
            memories,
            min_overlap=0.1,
            max_results=target_count * 2  # Get 2x, will refine further
        )

        # Step 5: If still too many, take most recent
        if len(memories) > target_count:
            memories = sorted(
                memories,
                key=lambda m: m.get("timestamp", ""),
                reverse=True
            )[:target_count]

        return memories

    def _infer_metadata_filters(self, query: str) -> Dict[str, Any]:
        """Infer metadata filters from query."""
        filters = {}

        # Detect language mentions
        languages = []
        if "python" in query.lower():
            languages.append("python")
        if "javascript" in query.lower() or "js" in query.lower():
            languages.append("javascript")
        if languages:
            filters["languages"] = languages

        # Detect if query mentions errors/bugs (prefer successful solutions)
        if any(word in query.lower() for word in ["fix", "bug", "error", "issue"]):
            filters["success_only"] = True

        return filters
```

**Estimated Impact**:
- Reduces memories to score: 1000 → 50 (95% reduction)
- API calls saved: 950 per query
- Cost reduction: ~$0.31 per task → $0.016 per task (95% savings)

---

## Phase 2: Multi-Level Caching (Week 2)

### Goal: Cache at multiple levels for maximum hit rate

### 2.1 Cache Level Architecture

**Implementation**: `memory_lib/caching/multi_level_cache.py`

```python
class CacheLevel(Enum):
    """Cache levels from most to least specific."""
    L1_EXACT = 1      # Exact context + memory match
    L2_SIMILAR = 2     # Similar context + same memory
    L3_STATIC = 3      # Memory features (context-independent)

@dataclass
class CachedScore:
    """Cached relevance score with metadata."""
    score: float
    reasoning: str
    cache_level: CacheLevel
    timestamp: float
    hit_count: int = 0

class MultiLevelCache:
    """Multi-level caching system for memory scores."""

    def __init__(
        self,
        l1_max_age: float = 3600,      # 1 hour
        l2_max_age: float = 86400,     # 24 hours
        l3_max_age: float = 604800     # 7 days
    ):
        self.l1_cache: Dict[str, CachedScore] = {}  # Exact matches
        self.l2_cache: Dict[str, CachedScore] = {}  # Similar context
        self.l3_cache: Dict[str, CachedScore] = {}  # Static features

        self.l1_max_age = l1_max_age
        self.l2_max_age = l2_max_age
        self.l3_max_age = l3_max_age

    def get(
        self,
        context: str,
        memory_id: str,
        memory_hash: str
    ) -> Optional[CachedScore]:
        """Try to retrieve score from cache (tries all levels)."""

        # L1: Exact match (context + memory + hash)
        l1_key = self._l1_key(context, memory_id, memory_hash)
        if l1_key in self.l1_cache:
            cached = self.l1_cache[l1_key]
            if self._is_fresh(cached.timestamp, self.l1_max_age):
                cached.hit_count += 1
                return cached

        # L2: Similar context (context features + memory)
        l2_key = self._l2_key(context, memory_id)
        if l2_key in self.l2_cache:
            cached = self.l2_cache[l2_key]
            if self._is_fresh(cached.timestamp, self.l2_max_age):
                cached.hit_count += 1
                # Adjust score slightly for similar (not exact) context
                return CachedScore(
                    score=cached.score * 0.95,  # Slight penalty
                    reasoning=f"[SIMILAR CONTEXT] {cached.reasoning}",
                    cache_level=CacheLevel.L2_SIMILAR,
                    timestamp=cached.timestamp,
                    hit_count=cached.hit_count
                )

        # L3: Static features (memory characteristics)
        l3_key = self._l3_key(memory_id)
        if l3_key in self.l3_cache:
            cached = self.l3_cache[l3_key]
            if self._is_fresh(cached.timestamp, self.l3_max_age):
                # L3 is context-independent, so more uncertain
                # Only use if score is extreme (very relevant or very irrelevant)
                if cached.score > 0.8 or cached.score < 0.3:
                    cached.hit_count += 1
                    return CachedScore(
                        score=cached.score * 0.9,  # Larger penalty
                        reasoning=f"[STATIC FEATURES] {cached.reasoning}",
                        cache_level=CacheLevel.L3_STATIC,
                        timestamp=cached.timestamp,
                        hit_count=cached.hit_count
                    )

        return None

    def set(
        self,
        context: str,
        memory_id: str,
        memory_hash: str,
        score: float,
        reasoning: str
    ):
        """Store score in all cache levels."""
        timestamp = time.time()

        # L1: Exact
        l1_key = self._l1_key(context, memory_id, memory_hash)
        self.l1_cache[l1_key] = CachedScore(
            score=score,
            reasoning=reasoning,
            cache_level=CacheLevel.L1_EXACT,
            timestamp=timestamp
        )

        # L2: Similar context
        l2_key = self._l2_key(context, memory_id)
        self.l2_cache[l2_key] = CachedScore(
            score=score,
            reasoning=reasoning,
            cache_level=CacheLevel.L2_SIMILAR,
            timestamp=timestamp
        )

        # L3: Static features
        l3_key = self._l3_key(memory_id)
        # Only update L3 if score is confident (extreme)
        if score > 0.7 or score < 0.4:
            self.l3_cache[l3_key] = CachedScore(
                score=score,
                reasoning=reasoning,
                cache_level=CacheLevel.L3_STATIC,
                timestamp=timestamp
            )

    def _l1_key(self, context: str, memory_id: str, memory_hash: str) -> str:
        """L1 cache key: exact match."""
        context_hash = hashlib.sha256(context.encode()).hexdigest()[:16]
        return f"l1:{context_hash}:{memory_id}:{memory_hash}"

    def _l2_key(self, context: str, memory_id: str) -> str:
        """L2 cache key: similar context."""
        # Extract context features (keywords, category)
        features = self._extract_context_features(context)
        features_hash = hashlib.sha256(features.encode()).hexdigest()[:16]
        return f"l2:{features_hash}:{memory_id}"

    def _l3_key(self, memory_id: str) -> str:
        """L3 cache key: memory only."""
        return f"l3:{memory_id}"

    def _extract_context_features(self, context: str) -> str:
        """Extract stable features from context for L2 caching."""
        # Remove variable parts (file names, line numbers, etc.)
        # Keep task type, keywords, category
        # This is a simplified version
        keywords = sorted(set(context.lower().split()))[:10]
        return " ".join(keywords)

    def _is_fresh(self, timestamp: float, max_age: float) -> bool:
        """Check if cache entry is still fresh."""
        return (time.time() - timestamp) < max_age

    def invalidate_memory(self, memory_id: str):
        """Invalidate all cache entries for a memory."""
        # Remove from all levels
        keys_to_remove = [
            k for k in self.l1_cache.keys()
            if memory_id in k
        ]
        for k in keys_to_remove:
            del self.l1_cache[k]

        keys_to_remove = [
            k for k in self.l2_cache.keys()
            if memory_id in k
        ]
        for k in keys_to_remove:
            del self.l2_cache[k]

        l3_key = self._l3_key(memory_id)
        if l3_key in self.l3_cache:
            del self.l3_cache[l3_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache),
            "l3_size": len(self.l3_cache),
            "total_entries": len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache),
            "l1_hit_rate": self._calculate_hit_rate(self.l1_cache),
            "l2_hit_rate": self._calculate_hit_rate(self.l2_cache),
            "l3_hit_rate": self._calculate_hit_rate(self.l3_cache)
        }

    def _calculate_hit_rate(self, cache: Dict) -> float:
        """Calculate hit rate for a cache level."""
        if not cache:
            return 0.0
        total_hits = sum(entry.hit_count for entry in cache.values())
        return total_hits / len(cache) if cache else 0.0
```

### 2.2 Persistent Cache Storage

**Implementation**: `memory_lib/caching/persistent_cache.py`

```python
class PersistentCache:
    """Persistent cache using SQLite for durability."""

    def __init__(self, db_path: str = "memory_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS score_cache (
                    cache_key TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    reasoning TEXT,
                    cache_level INTEGER,
                    timestamp REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    INDEX idx_memory_id (memory_id),
                    INDEX idx_timestamp (timestamp)
                )
            """)
            conn.commit()

    def load_to_memory(self) -> MultiLevelCache:
        """Load cache from disk to memory."""
        cache = MultiLevelCache()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM score_cache")

            for row in cursor:
                cached_score = CachedScore(
                    score=row["score"],
                    reasoning=row["reasoning"],
                    cache_level=CacheLevel(row["cache_level"]),
                    timestamp=row["timestamp"],
                    hit_count=row["hit_count"]
                )

                # Route to appropriate cache level
                if cached_score.cache_level == CacheLevel.L1_EXACT:
                    cache.l1_cache[row["cache_key"]] = cached_score
                elif cached_score.cache_level == CacheLevel.L2_SIMILAR:
                    cache.l2_cache[row["cache_key"]] = cached_score
                elif cached_score.cache_level == CacheLevel.L3_STATIC:
                    cache.l3_cache[row["cache_key"]] = cached_score

        return cache

    def save_from_memory(self, cache: MultiLevelCache):
        """Save memory cache to disk."""
        with sqlite3.connect(self.db_path) as conn:
            # Clear old entries
            conn.execute("DELETE FROM score_cache")

            # Save all cache levels
            all_entries = []

            for key, entry in cache.l1_cache.items():
                all_entries.append((
                    key,
                    self._extract_memory_id(key),
                    entry.score,
                    entry.reasoning,
                    entry.cache_level.value,
                    entry.timestamp,
                    entry.hit_count
                ))

            # Same for L2 and L3...

            conn.executemany(
                """INSERT INTO score_cache
                   (cache_key, memory_id, score, reasoning, cache_level, timestamp, hit_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                all_entries
            )
            conn.commit()
```

**Estimated Impact**:
- L1 hit rate: 60-70% (same query repeated)
- L2 hit rate: 20-30% (similar queries)
- L3 hit rate: 10-15% (general memory usefulness)
- **Combined hit rate: 90-95%**
- **Cost reduction: 90-95% on repeated queries**

---

## Phase 3: Integration (Week 3)

### 3.1 Enhanced Code Memory Retrieval

**Update**: `memory_lib/codebase/code_retrieval.py`

```python
class EnhancedCodeMemoryRetrieval(CodeMemoryRetrieval):
    """Enhanced retrieval with smart indexing and multi-level caching."""

    def __init__(
        self,
        small_model_fn,
        smart_index: SmartMemoryIndex = None,
        multi_level_cache: MultiLevelCache = None,
        **kwargs
    ):
        super().__init__(small_model_fn, **kwargs)

        self.smart_index = smart_index or SmartMemoryIndex()
        self.cache = multi_level_cache or MultiLevelCache()

        # Track statistics
        self.stats = {
            "total_queries": 0,
            "memories_pre_filtered": 0,
            "cache_hits": 0,
            "api_calls": 0
        }

    async def retrieve_code_memories(
        self,
        code_context: CodeContext,
        memories: List[Dict[str, Any]],
        file_hashes: Optional[Dict[str, str]] = None
    ) -> List[ScoredMemory]:
        """Enhanced retrieval with smart indexing and caching."""
        self.stats["total_queries"] += 1

        # PHASE 1: Smart Pre-Filtering (fast, no API calls)
        context_str = code_context.to_context_string()
        pre_filtered = self.smart_index.pre_filter(
            context_str,
            memories,
            target_count=50  # Reduce from 1000 to 50
        )

        self.stats["memories_pre_filtered"] += (len(memories) - len(pre_filtered))

        # PHASE 2: Score with Multi-Level Caching
        scored_memories = []
        needs_scoring = []

        for memory in pre_filtered:
            memory_id = memory["id"]
            memory_hash = file_hashes.get(memory.get("file_path", ""), "")

            # Try cache first
            cached = self.cache.get(context_str, memory_id, memory_hash)

            if cached:
                # Cache hit!
                self.stats["cache_hits"] += 1
                scored_memories.append(ScoredMemory(
                    memory_id=memory_id,
                    text=self._format_memory_text(memory),
                    metadata=memory.get("metadata", {}),
                    timestamp=memory.get("timestamp", ""),
                    relevance_score=cached.score,
                    reasoning=cached.reasoning
                ))
            else:
                # Need to score with LLM
                needs_scoring.append(memory)

        # PHASE 3: Score remaining memories with LLM (parallel)
        if needs_scoring:
            self.stats["api_calls"] += len(needs_scoring)

            newly_scored = []
            for i in range(0, len(needs_scoring), self.batch_size):
                batch = needs_scoring[i:i + self.batch_size]
                tasks = [
                    self._score_code_memory_enhanced(
                        code_context,
                        mem,
                        file_hashes
                    )
                    for mem in batch
                ]
                batch_scored = await asyncio.gather(*tasks)
                newly_scored.extend(batch_scored)

            scored_memories.extend(newly_scored)

        # PHASE 4: Apply other optimizations (dependency boost, recency)
        scored_memories = self._apply_dependency_boost(scored_memories, memories)
        scored_memories = self._apply_recency_boost(scored_memories, memories)

        # PHASE 5: Filter and select top-K
        selected = self.filter_and_select(scored_memories)

        return selected

    async def _score_code_memory_enhanced(
        self,
        code_context: CodeContext,
        memory: Dict[str, Any],
        file_hashes: Dict[str, str]
    ) -> ScoredMemory:
        """Score a memory and cache the result."""
        context_str = code_context.to_context_string()
        memory_id = memory["id"]
        memory_hash = file_hashes.get(memory.get("file_path", ""), "")

        # Score with LLM
        prompt = self._create_code_scoring_prompt(code_context, memory)

        try:
            response = await self.small_model_fn(prompt)
            score, reasoning = self._parse_score_response(response)

            # Cache the result in all levels
            self.cache.set(context_str, memory_id, memory_hash, score, reasoning)

            return ScoredMemory(
                memory_id=memory_id,
                text=self._format_memory_text(memory),
                metadata=memory.get("metadata", {}),
                timestamp=memory.get("timestamp", ""),
                relevance_score=score,
                reasoning=reasoning
            )
        except Exception as e:
            # Error, return low score
            return ScoredMemory(
                memory_id=memory_id,
                text=self._format_memory_text(memory),
                metadata=memory.get("metadata", {}),
                timestamp=memory.get("timestamp", ""),
                relevance_score=0.0,
                reasoning=f"Error: {str(e)}"
            )

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics on optimizations."""
        total_queries = self.stats["total_queries"]
        cache_hits = self.stats["cache_hits"]
        api_calls = self.stats["api_calls"]

        return {
            "total_queries": total_queries,
            "memories_pre_filtered": self.stats["memories_pre_filtered"],
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / (cache_hits + api_calls) if (cache_hits + api_calls) > 0 else 0,
            "api_calls": api_calls,
            "api_calls_per_query": api_calls / total_queries if total_queries > 0 else 0,
            "estimated_cost_savings": self._estimate_cost_savings()
        }

    def _estimate_cost_savings(self) -> Dict[str, float]:
        """Estimate cost savings from optimizations."""
        # Assume baseline: 1000 memories per query, all scored
        baseline_calls = self.stats["total_queries"] * 1000
        actual_calls = self.stats["api_calls"]

        # Assume $0.0005 per 1K tokens, avg 550 tokens per call
        cost_per_call = 0.0005 * 0.55

        baseline_cost = baseline_calls * cost_per_call
        actual_cost = actual_calls * cost_per_call
        savings = baseline_cost - actual_cost

        return {
            "baseline_cost": baseline_cost,
            "actual_cost": actual_cost,
            "savings": savings,
            "savings_percentage": (savings / baseline_cost * 100) if baseline_cost > 0 else 0
        }
```

### 3.2 Update CodeMemorySystem

**Update**: `memory_lib/codebase/code_memory_system.py`

```python
class CodeMemorySystem:
    """Enhanced code memory system with smart indexing."""

    def __init__(self, small_model_fn, **kwargs):
        self.storage = CodeMemoryStorage(kwargs.get("db_path"))

        # Initialize smart indexing
        self.smart_index = SmartMemoryIndex()

        # Initialize multi-level caching
        self.persistent_cache = PersistentCache(
            db_path=kwargs.get("cache_db_path", "memory_cache.db")
        )
        self.multi_level_cache = self.persistent_cache.load_to_memory()

        # Initialize enhanced retrieval
        self.retrieval = EnhancedCodeMemoryRetrieval(
            small_model_fn=small_model_fn,
            smart_index=self.smart_index,
            multi_level_cache=self.multi_level_cache,
            **kwargs
        )

        self.indexer = CodeIndexer()

    def save_cache(self):
        """Save cache to disk."""
        self.persistent_cache.save_from_memory(self.multi_level_cache)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        retrieval_stats = self.retrieval.get_optimization_stats()
        cache_stats = self.multi_level_cache.get_stats()

        return {
            "retrieval": retrieval_stats,
            "cache": cache_stats
        }
```

---

## Phase 4: Terminal-Bench Agent Integration (Week 3)

### 4.1 Update Agent to Use Enhanced System

**Update**: `terminal_bench_agent/core.py`

```python
class MemoryGuidedAgent(BaseAgent):
    """Agent with enhanced memory system."""

    def __init__(self, llm_function, memory_system=None, **kwargs):
        # Memory system should now be EnhancedCodeMemorySystem
        self.memory = memory_system
        # ... rest of init

    async def _solve_task_async(self, task_description: str, logging_dir):
        # ... existing code ...

        # After task completion, save cache
        if self.memory:
            self.memory.save_cache()

            # Log optimization stats
            if logging_dir:
                stats = self.memory.get_optimization_stats()
                self._log(logging_dir, f"\nOptimization Stats:\n{json.dumps(stats, indent=2)}\n")
```

### 4.2 Configuration Updates

**Update**: `terminal_bench_agent/config/agent_config.yaml`

```yaml
memory:
  enabled: true
  db_path: "terminal_bench_memories.db"
  cache_db_path: "terminal_bench_cache.db"

  # Retrieval settings
  relevance_threshold: 0.7
  max_memories: 10

  # Smart indexing
  smart_indexing:
    enabled: true
    pre_filter_target: 50  # Reduce 1000 memories to 50
    min_keyword_overlap: 0.1

  # Multi-level caching
  caching:
    enabled: true
    l1_max_age_seconds: 3600    # 1 hour
    l2_max_age_seconds: 86400   # 24 hours
    l3_max_age_seconds: 604800  # 7 days
    persist_to_disk: true

  # Optimizations
  enable_dependency_boost: true
  enable_recency_boost: true
```

---

## Phase 5: Monitoring & Analytics (Week 4)

### 5.1 Cost Tracking Dashboard

**Implementation**: `memory_lib/monitoring/cost_tracker.py`

```python
class CostTracker:
    """Track and report memory system costs."""

    def __init__(
        self,
        small_model_price_per_1k: float = 0.0005,
        primary_model_price_per_1k: float = 0.015
    ):
        self.small_model_price = small_model_price_per_1k
        self.primary_model_price = primary_model_price_per_1k

        self.calls = {
            "small_model": 0,
            "primary_model": 0
        }

        self.tokens = {
            "small_model_input": 0,
            "small_model_output": 0,
            "primary_model_input": 0,
            "primary_model_output": 0
        }

    def record_small_model_call(self, input_tokens: int, output_tokens: int):
        """Record a small model API call."""
        self.calls["small_model"] += 1
        self.tokens["small_model_input"] += input_tokens
        self.tokens["small_model_output"] += output_tokens

    def record_primary_model_call(self, input_tokens: int, output_tokens: int):
        """Record a primary model API call."""
        self.calls["primary_model"] += 1
        self.tokens["primary_model_input"] += input_tokens
        self.tokens["primary_model_output"] += output_tokens

    def get_total_cost(self) -> float:
        """Calculate total cost."""
        small_cost = (
            (self.tokens["small_model_input"] + self.tokens["small_model_output"]) / 1000
            * self.small_model_price
        )

        primary_cost = (
            (self.tokens["primary_model_input"] + self.tokens["primary_model_output"]) / 1000
            * self.primary_model_price
        )

        return small_cost + primary_cost

    def get_report(self) -> Dict[str, Any]:
        """Get detailed cost report."""
        return {
            "calls": self.calls,
            "tokens": self.tokens,
            "costs": {
                "small_model": (
                    (self.tokens["small_model_input"] + self.tokens["small_model_output"]) / 1000
                    * self.small_model_price
                ),
                "primary_model": (
                    (self.tokens["primary_model_input"] + self.tokens["primary_model_output"]) / 1000
                    * self.primary_model_price
                ),
                "total": self.get_total_cost()
            }
        }
```

### 5.2 Analytics Dashboard

**Implementation**: `scripts/view_analytics.py`

```python
#!/usr/bin/env python3
"""View memory system analytics and cost savings."""

import json
from memory_lib import CodeMemorySystem

def show_analytics(memory_system: CodeMemorySystem):
    """Show comprehensive analytics."""
    stats = memory_system.get_optimization_stats()

    print("=" * 70)
    print("Memory System Optimization Analytics")
    print("=" * 70)

    print("\n📊 Retrieval Statistics:")
    print(f"  Total Queries: {stats['retrieval']['total_queries']}")
    print(f"  Memories Pre-filtered: {stats['retrieval']['memories_pre_filtered']}")
    print(f"  Cache Hits: {stats['retrieval']['cache_hits']}")
    print(f"  Cache Hit Rate: {stats['retrieval']['cache_hit_rate']:.1%}")
    print(f"  API Calls: {stats['retrieval']['api_calls']}")
    print(f"  API Calls per Query: {stats['retrieval']['api_calls_per_query']:.1f}")

    print("\n💰 Cost Savings:")
    savings = stats['retrieval']['estimated_cost_savings']
    print(f"  Baseline Cost: ${savings['baseline_cost']:.2f}")
    print(f"  Actual Cost: ${savings['actual_cost']:.2f}")
    print(f"  Savings: ${savings['savings']:.2f} ({savings['savings_percentage']:.1f}%)")

    print("\n🗄️  Cache Statistics:")
    print(f"  L1 Cache Size: {stats['cache']['l1_size']}")
    print(f"  L2 Cache Size: {stats['cache']['l2_size']}")
    print(f"  L3 Cache Size: {stats['cache']['l3_size']}")
    print(f"  L1 Hit Rate: {stats['cache']['l1_hit_rate']:.1%}")
    print(f"  L2 Hit Rate: {stats['cache']['l2_hit_rate']:.1%}")
    print(f"  L3 Hit Rate: {stats['cache']['l3_hit_rate']:.1%}")

    print("\n" + "=" * 70)
```

---

## Implementation Schedule

### Week 1: Smart Indexing
- **Day 1-2**: Task categorization system
- **Day 3-4**: Keyword and metadata filtering
- **Day 5-7**: Combined smart index, testing

### Week 2: Multi-Level Caching
- **Day 1-3**: Multi-level cache implementation
- **Day 4-5**: Persistent cache storage
- **Day 6-7**: Integration with retrieval

### Week 3: Integration
- **Day 1-2**: Enhanced retrieval integration
- **Day 3-4**: Terminal-Bench agent updates
- **Day 5-7**: End-to-end testing

### Week 4: Monitoring
- **Day 1-2**: Cost tracking
- **Day 3-4**: Analytics dashboard
- **Day 5-7**: Documentation and polish

## Success Metrics

### Target Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memories scored per query | 1000 | 50 | 95% reduction |
| API calls per query | 1000 | 5-10 | 99% reduction |
| Cost per task | $0.62 | $0.03-0.06 | 90-95% reduction |
| Cache hit rate | 0% | 90%+ | - |
| Query latency | 50s | 2-5s | 90% reduction |

### Phase 1 Success Criteria
- ✅ Pre-filter reduces memories by 90%+
- ✅ Keyword matching accuracy > 80%
- ✅ Category detection accuracy > 70%

### Phase 2 Success Criteria
- ✅ Cache hit rate > 80%
- ✅ L1 hit rate > 60%
- ✅ Cache persists across sessions

### Phase 3 Success Criteria
- ✅ Full integration with no breaking changes
- ✅ Cost reduction > 85%
- ✅ Retrieval quality maintained or improved

### Phase 4 Success Criteria
- ✅ Cost tracking accurate within 5%
- ✅ Analytics dashboard functional
- ✅ Documentation complete

## Testing Strategy

### Unit Tests
```python
# Test smart indexing
def test_task_categorization():
    categorizer = TaskCategorizer()
    assert categorizer.categorize("ping google.com") == TaskCategory.NETWORK

def test_keyword_filtering():
    filter = KeywordFilter()
    memories = filter.filter_memories("python script", all_memories, max_results=50)
    assert len(memories) <= 50

# Test caching
def test_cache_levels():
    cache = MultiLevelCache()
    cache.set("context", "mem1", "hash1", 0.9, "relevant")
    cached = cache.get("context", "mem1", "hash1")
    assert cached.score == 0.9
    assert cached.cache_level == CacheLevel.L1_EXACT
```

### Integration Tests
```python
# Test full pipeline
@pytest.mark.asyncio
async def test_enhanced_retrieval():
    retrieval = EnhancedCodeMemoryRetrieval(...)
    memories = await retrieval.retrieve_code_memories(context, all_memories)

    # Verify pre-filtering worked
    assert retrieval.stats["memories_pre_filtered"] > 900

    # Verify caching worked
    assert retrieval.stats["cache_hits"] > 0

    # Verify quality maintained
    assert all(m.relevance_score >= 0.7 for m in memories)
```

### Benchmark Tests
```bash
# Compare before/after on 10 tasks
python scripts/benchmark_optimizations.py --tasks 10

# Expected output:
# Before: 1000 API calls, $6.20, 8min
# After: 50 API calls, $0.31, 30sec
# Savings: 95% cost, 94% time
```

## Risk Mitigation

### Risk 1: Cache Staleness
**Mitigation**:
- Time-based expiration (L1: 1hr, L2: 24hr, L3: 7d)
- Invalidate on memory updates
- Monitor cache hit quality

### Risk 2: Pre-filtering Too Aggressive
**Mitigation**:
- Fallback to full retrieval if pre-filter returns < 10 memories
- Tune min_keyword_overlap threshold
- Category-based failsafe

### Risk 3: Cache Memory Usage
**Mitigation**:
- LRU eviction when cache size > threshold
- Persist to disk
- Clear old entries periodically

### Risk 4: Reduced Retrieval Quality
**Mitigation**:
- A/B test on subset of tasks
- Monitor success rate changes
- Adjustable pre-filter thresholds

## Monitoring Plan

### Daily Metrics
- Cache hit rates (L1, L2, L3)
- API calls per query
- Cost per task
- Query latency

### Weekly Review
- Retrieval quality (success rate)
- Cache size growth
- Cost trends
- Anomaly detection

### Alerts
- Cache hit rate < 70% → investigate
- Cost per task > $0.10 → investigate
- Query latency > 10s → investigate

## Rollout Plan

### Phase 1: Development (Week 1-3)
- Implement features
- Unit testing
- Integration testing

### Phase 2: Alpha Testing (Week 4)
- Deploy to dev environment
- Run on 20 tasks
- Monitor metrics
- Fix issues

### Phase 3: Beta Testing (Week 5)
- Deploy to staging
- Run on 50 tasks
- Compare with baseline
- Tune parameters

### Phase 4: Production (Week 6)
- Deploy to production
- Gradual rollout (10% → 50% → 100%)
- Monitor closely
- Rollback plan ready

## Documentation Updates

### User Documentation
- Update README.md with optimization details
- Add configuration guide for caching
- Add cost estimation guide

### Developer Documentation
- API documentation for new components
- Architecture diagrams
- Contributing guide for optimizations

## Cost-Benefit Analysis

### Investment
- Development time: 4 weeks
- Testing time: 1-2 weeks
- **Total**: 5-6 weeks

### Returns
- Cost savings: 90-95% per task
- Latency reduction: 90%
- Scalability: 10x more memories without cost increase
- **ROI**: Break-even after ~100 benchmark runs

### Long-term Benefits
- Scales to 10,000+ memories
- Enables continuous learning
- Reduces environmental impact (fewer API calls)
- Improves user experience (faster responses)

---

## Summary

This plan implements two key optimizations:

1. **Smart Indexing** (95% reduction in memories to score)
   - Task categorization
   - Keyword filtering
   - Metadata filtering
   - Target: 1000 → 50 memories

2. **Multi-Level Caching** (90%+ cache hit rate)
   - L1: Exact matches
   - L2: Similar contexts
   - L3: Static features
   - Persistent storage

**Expected Results**:
- API calls: 1000 → 5-10 per query (99% reduction)
- Cost: $0.62 → $0.03-0.06 per task (90-95% reduction)
- Latency: 50s → 2-5s (90% reduction)
- **Full benchmark: $62 → $3-6 (95% savings)**

The system maintains or improves retrieval quality while dramatically reducing costs and latency!
