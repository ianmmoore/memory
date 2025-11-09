## Complete Memory System Architecture

### General Version

**Storage Layer**
- Store memories as structured entries: `{text, metadata, timestamp, id}`
- Use simple database (SQL, NoSQL) for persistent storage
- Maintain indices for efficient iteration

**Retrieval Pipeline** (per turn)

1. **Exhaustive Reasoning via API** (small model)
   - Fetch all N memories from database
   - Construct N prompts: `"Context: {current_context}\nMemory: {memory_i}\nIs this memory relevant? Score 0-1 and explain briefly."`
   - Batch and parallelize API calls to small model
   - Parse responses to extract `{memory_id, relevance_score, reason}`

2. **Filtering & Selection**
   - Filter: keep only memories where `relevance_score >= threshold` (e.g., 0.7)
   - If filtered set has ≤ K memories: use all of them
   - If filtered set has > K memories: take top-K by score
   - Result: high-quality memories, bounded by K for context length

3. **Primary Model Call**
   - Construct prompt: `<memory id=X score=S>{content}</memory>...` + current task
   - Single API call to primary model
   - Return response

**Implementation Details**
- Async/await for parallel small model API calls
- Retry logic with exponential backoff
- Monitor costs: N × small_model_cost + 1 × large_model_cost per turn

---

### Codebase-Specific Version

**Storage Layer**
- Code memories: `{file_path, entity_name, code_snippet, docstring, signature, timestamp, id}`
- Granularity: functions, classes, important blocks
- Metadata: `{language, dependencies, imports, complexity, last_modified}`
- Non-code memories: README sections, architecture docs, past debugging sessions, design decisions

**What Gets Indexed**
- Every function/class definition in repository
- Configuration files with explanations
- Previous conversations about code sections
- Error patterns and resolutions
- API documentation and usage examples
- Test cases and their purposes

**Retrieval Pipeline** (per coding turn)

1. **Context Construction**
   - Current context includes:
     - User query/task
     - Current file being edited
     - Recent code changes
     - Active error messages or stack traces
     - Recently accessed files

2. **Exhaustive Reasoning via API** (small model)
   - For each code memory, construct prompt:
   ```
   Task: {user_query}
   Current context: {current_file, recent_changes, errors}
   
   Code memory:
   File: {file_path}
   Function: {entity_name}
   Code: {code_snippet}
   
   Rate relevance 0-1. Consider:
   - Direct dependencies
   - Similar patterns/logic
   - Related functionality
   - Error handling for current issue
   ```
   - Parallel API calls across all code memories
   - Parse `{memory_id, relevance_score, reasoning}`

3. **Filtering & Selection**
   - Filter: keep memories where `relevance_score >= 0.7`
   - Sort by relevance_score descending
   - If filtered set has ≤ K memories (K=15 for code): use all
   - If filtered set has > K memories: take top-K
   - Result: highly relevant code context, bounded for token limits

4. **Primary Model Call**
   - Prompt structure:
   ```
   Relevant code context:
   <code file="{path}" entity="{name}" score="{score}">
   {code_snippet}
   </code>
   ...
   
   Current file: {file_being_edited}
   Task: {user_query}
   Generate solution with full context.
   ```
   - Single API call to primary model

**Codebase Optimizations**
- **Caching**: Cache relevance scores for unchanged code; invalidate on file modification
- **Dependency awareness**: If function A is relevant and calls B, automatically boost B's score
- **Recency weighting**: Recently modified files get slight score boost
- **File-level grouping**: If multiple entities from same file are relevant, include file header/imports once
- **Test-code linking**: When editing code, automatically include corresponding test files in memory pool

**Cost Management**
- Cache small model responses for unchanged code
- Incremental updates: only re-score memories when their files change
- Batch user requests before re-evaluating entire memory pool
