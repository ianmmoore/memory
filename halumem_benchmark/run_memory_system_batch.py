#!/usr/bin/env python3
"""Run HaluMem benchmark using our MemorySystem with Batch API.

Uses Batch API for ALL LLM calls:
- gpt-5.1: extraction, updates, QA answers, judging
- gpt-5-nano: relevance scoring (batched with separate chunking)

This gives us:
- No rate limiting on any LLM calls
- 50% cost savings on all API calls
- Full retrieval pipeline with embedding prefilter + LLM scoring
- Resumable execution (exit and resume later)

Usage:
    python run_memory_system_batch.py
    python run_memory_system_batch.py --sample-size 50
    python run_memory_system_batch.py --resume results/memsys_batch_long_20241123_120000
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from memory_lib.general import MemorySystem, create_openai_embedding_fn
from halumem_benchmark import HaluMemConfig, HaluMemDataset
from halumem_benchmark.batch_api import BatchProcessor
from halumem_benchmark.metrics import (
    compute_extraction_metrics,
    compute_update_metrics,
    compute_qa_metrics
)

logger = logging.getLogger(__name__)

# Token limit for batch API (leaving buffer under 900K limit)
MAX_BATCH_TOKENS = 800000

# gpt-5-nano is cheaper, can use larger chunks
MAX_NANO_BATCH_TOKENS = 800000


def create_scoring_prompt(context: str, memory_text: str, metadata: dict = None) -> str:
    """Create a prompt for relevance scoring."""
    metadata_str = ", ".join(f"{k}: {v}" for k, v in (metadata or {}).items()) or "none"
    return f"""Context: {context}

Memory: {memory_text}
Metadata: {metadata_str}

Is this memory relevant to the current context? Rate relevance from 0 to 1 (0 = not relevant, 1 = highly relevant).
Provide your response in this exact format:
Score: <number between 0 and 1>
Reason: <brief explanation>"""


def parse_score_response(response: str) -> float:
    """Parse the scoring response to extract the score."""
    import re
    try:
        score_match = re.search(r"Score:\s*([\d.]+)", response, re.IGNORECASE)
        if not score_match:
            return 0.0
        score = float(score_match.group(1))
        return max(0.0, min(1.0, score))
    except:
        return 0.0


async def batch_score_memories(
    batch_processor,
    queries_and_memories: List[Dict[str, Any]],
    state: Dict[str, Any],
    state_key: str,
    run_dir: Path,
    threshold: float = 0.5,
    max_per_query: int = 20
) -> Dict[str, List[Dict[str, Any]]]:
    """Batch score memories for multiple queries using gpt-5-nano.

    Args:
        batch_processor: BatchProcessor instance
        queries_and_memories: List of {"query_id": str, "query": str, "memories": [{"id", "text", "metadata"}]}
        state: State dict for resume support
        state_key: Key for tracking this batch
        run_dir: Directory for state
        threshold: Minimum score to include (0-1)
        max_per_query: Maximum memories to return per query

    Returns:
        Dict mapping query_id to list of scored memories
    """
    # Build all scoring requests
    scoring_requests = []
    request_map = {}  # custom_id -> (query_id, memory_id)

    for item in queries_and_memories:
        query_id = item["query_id"]
        query = item["query"]

        for mem in item["memories"]:
            custom_id = f"score_{query_id}_{mem['id']}"
            prompt = create_scoring_prompt(query, mem["text"], mem.get("metadata"))

            # Note: gpt-5-nano only supports temperature=1 (default), cannot use 0
            scoring_requests.append(batch_processor._create_batch_request(
                custom_id=custom_id,
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}]
            ))
            request_map[custom_id] = (query_id, mem["id"], mem["text"], mem.get("metadata", {}))

    if not scoring_requests:
        return {}

    # Chunk and process
    scoring_chunks = chunk_requests(scoring_requests, MAX_NANO_BATCH_TOKENS)
    logger.info(f"Processing {len(scoring_requests)} scoring requests in {len(scoring_chunks)} chunks")

    scoring_responses = await process_chunked_batch(
        batch_processor=batch_processor,
        chunks=scoring_chunks,
        description="Relevance scoring - gpt-5-nano",
        state=state,
        state_key=state_key,
        run_dir=run_dir
    )

    # Parse scores and group by query
    query_scores = {}  # query_id -> [(score, memory_dict)]

    for custom_id, response in scoring_responses.items():
        if custom_id not in request_map:
            continue

        query_id, mem_id, mem_text, mem_metadata = request_map[custom_id]
        score = parse_score_response(response)

        if query_id not in query_scores:
            query_scores[query_id] = []

        if score >= threshold:
            query_scores[query_id].append({
                "memory_id": mem_id,
                "text": mem_text,
                "metadata": mem_metadata,
                "score": score
            })

    # Sort by score and limit per query
    results = {}
    for query_id, memories in query_scores.items():
        sorted_mems = sorted(memories, key=lambda x: x["score"], reverse=True)
        results[query_id] = sorted_mems[:max_per_query]

    return results


async def bulk_embedding_prefilter(
    embed_fn,
    queries: List[Dict[str, str]],
    all_memories: List[Dict[str, Any]],
    top_k: int = 100,
    memory_system=None
) -> Dict[str, List[Dict[str, Any]]]:
    """Prefilter memories for multiple queries using embeddings.

    Memory-efficient version that processes queries in batches.

    Args:
        embed_fn: Async function to generate embeddings
        queries: List of {"query_id": str, "query": str}
        all_memories: List of all memories with embeddings (can be None if memory_system provided)
        top_k: Number of top candidates to return per query
        memory_system: Optional MemorySystem to load embeddings more efficiently

    Returns:
        Dict mapping query_id to list of candidate memories
    """
    import numpy as np
    import gc

    if not queries:
        return {}

    # Load embeddings efficiently if memory_system is provided
    if memory_system is not None:
        logger.info("Loading embeddings from database (memory-efficient mode)...")
        memory_ids, memory_embeddings = memory_system.storage.get_all_embeddings_as_array()
        if len(memory_ids) == 0:
            logger.warning("No embeddings found, returning empty results")
            return {}
        logger.info(f"Loaded {len(memory_ids)} embeddings, shape: {memory_embeddings.shape}")
    else:
        # Fallback to old method
        if not all_memories:
            return {}
        memories_with_embeddings = [m for m in all_memories if m.get("embedding")]
        if not memories_with_embeddings:
            logger.warning("No embeddings found, returning empty results")
            return {}
        memory_ids = [m["id"] for m in memories_with_embeddings]
        memory_embeddings = np.array([m["embedding"] for m in memories_with_embeddings], dtype=np.float32)

    # Pre-normalize memory embeddings (do this once)
    logger.info("Normalizing memory embeddings...")
    norms = np.linalg.norm(memory_embeddings, axis=1, keepdims=True)
    memory_embeddings_norm = memory_embeddings / (norms + 1e-9)
    del norms
    gc.collect()

    # Process queries in batches to avoid memory issues
    query_batch_size = 200  # Process 200 queries at a time
    embed_batch_size = 100  # Embed 100 texts at a time
    results = {}

    for batch_start in range(0, len(queries), query_batch_size):
        batch_end = min(batch_start + query_batch_size, len(queries))
        query_batch = queries[batch_start:batch_end]

        logger.info(f"Processing queries {batch_start+1}-{batch_end} of {len(queries)}...")

        # Get embeddings for this batch of queries
        query_texts = [q["query"] for q in query_batch]
        query_embeddings = []
        for i in range(0, len(query_texts), embed_batch_size):
            batch = query_texts[i:i + embed_batch_size]
            batch_embeddings = await embed_fn(batch)
            query_embeddings.extend(batch_embeddings)

        # Convert to numpy and normalize
        query_embeddings = np.array(query_embeddings, dtype=np.float32)
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings_norm = query_embeddings / (query_norms + 1e-9)
        del query_embeddings, query_norms
        gc.collect()

        # Compute similarities: (num_queries x embedding_dim) @ (embedding_dim x num_memories)
        # Result: (num_queries x num_memories)
        similarities = np.dot(query_embeddings_norm, memory_embeddings_norm.T)
        del query_embeddings_norm
        gc.collect()

        # Extract top-k for each query in this batch
        for i, query in enumerate(query_batch):
            query_sims = similarities[i]

            # Get top-k indices efficiently
            if len(query_sims) <= top_k:
                top_indices = np.argsort(query_sims)[::-1]
            else:
                top_indices = np.argpartition(query_sims, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(query_sims[top_indices])[::-1]]

            # Store just IDs and scores for now
            results[query["query_id"]] = [
                {"id": memory_ids[idx], "embedding_score": float(query_sims[idx])}
                for idx in top_indices
            ]

        del similarities
        gc.collect()

    # Now fetch full memory details for top candidates
    logger.info("Fetching memory details for candidates...")
    all_needed_ids = set()
    for candidates in results.values():
        all_needed_ids.update(c["id"] for c in candidates)

    # Fetch from database in batches
    if memory_system is not None:
        id_to_memory = {}
        id_list = list(all_needed_ids)
        fetch_batch_size = 1000
        for i in range(0, len(id_list), fetch_batch_size):
            batch_ids = id_list[i:i + fetch_batch_size]
            batch_memories = memory_system.storage.get_memories_by_ids(batch_ids)
            for m in batch_memories:
                id_to_memory[m["id"]] = m
    else:
        id_to_memory = {m["id"]: m for m in all_memories if m["id"] in all_needed_ids}

    # Build final results with full memory info
    final_results = {}
    for query_id, candidates in results.items():
        final_candidates = []
        for c in candidates:
            mem = id_to_memory.get(c["id"])
            if mem:
                final_candidates.append({
                    "id": mem["id"],
                    "text": mem["text"],
                    "metadata": mem.get("metadata", {}),
                    "embedding_score": c["embedding_score"]
                })
        final_results[query_id] = final_candidates

    del results, id_to_memory
    gc.collect()

    return final_results


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: ~3 chars per token."""
    return len(text) // 3


def chunk_requests(
    requests: List[Dict[str, Any]],
    max_tokens: int = MAX_BATCH_TOKENS
) -> List[List[Dict[str, Any]]]:
    """Split requests into chunks that fit under token limit.

    Args:
        requests: List of batch request dicts
        max_tokens: Maximum tokens per chunk

    Returns:
        List of request chunks
    """
    if not requests:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0

    for req in requests:
        # Estimate tokens from the request content
        req_tokens = estimate_tokens(json.dumps(req))

        # If adding this request would exceed limit, start new chunk
        if current_tokens + req_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(req)
        current_tokens += req_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    logger.info(f"Split {len(requests)} requests into {len(chunks)} chunks")
    return chunks


async def process_chunked_batch(
    batch_processor,
    chunks: List[List[Dict[str, Any]]],
    description: str,
    state: Dict[str, Any],
    state_key: str,
    run_dir: Path
) -> Dict[str, str]:
    """Process multiple chunks sequentially, with resume support.

    Args:
        batch_processor: BatchProcessor instance
        chunks: List of request chunks
        description: Description for batch jobs
        state: Current state dict
        state_key: Key in state for tracking this batch set (e.g., 'extraction_chunks')
        run_dir: Directory for saving state

    Returns:
        Combined results from all chunks
    """
    # Initialize chunk tracking if not present or if length doesn't match
    # (handles case where state was reset to [] but new chunks were created)
    if state_key not in state or len(state.get(state_key, [])) != len(chunks):
        state[state_key] = [
            {"chunk_idx": i, "batch_id": None, "status": "pending"}
            for i in range(len(chunks))
        ]
        save_state(run_dir, state)

    chunk_states = state[state_key]
    all_results = {}

    # Load any previously saved partial results
    partial_results_file = run_dir / f"{state_key}_partial.json"
    if partial_results_file.exists():
        with open(partial_results_file) as f:
            all_results = json.load(f)
        logger.info(f"Loaded {len(all_results)} partial results from previous run")

    for i, chunk in enumerate(chunks):
        chunk_state = chunk_states[i]

        if chunk_state["status"] == "completed":
            logger.info(f"Chunk {i+1}/{len(chunks)} already completed, skipping")
            continue

        if chunk_state["batch_id"] and chunk_state["status"] == "in_progress":
            # Resume polling existing batch
            logger.info(f"Resuming chunk {i+1}/{len(chunks)}: {chunk_state['batch_id']}")
            batch_id = chunk_state["batch_id"]
        else:
            # Submit new batch
            logger.info(f"Submitting chunk {i+1}/{len(chunks)} ({len(chunk)} requests)...")
            batch = await batch_processor.submit_batch(
                chunk, f"{description} - chunk {i+1}/{len(chunks)}"
            )
            batch_id = batch.batch_id

            # Update state
            chunk_state["batch_id"] = batch_id
            chunk_state["status"] = "in_progress"
            save_state(run_dir, state)
            logger.info(f"Chunk {i+1} submitted as {batch_id}. Safe to exit - will resume.")

        # Wait for completion
        logger.info(f"Waiting for chunk {i+1}/{len(chunks)} ({batch_id})...")
        chunk_results = await batch_processor.wait_for_completion(batch_id)

        # Merge results
        all_results.update(chunk_results)

        # Mark complete and save partial results
        chunk_state["status"] = "completed"
        save_state(run_dir, state)

        with open(partial_results_file, 'w') as f:
            json.dump(all_results, f)

        logger.info(f"Chunk {i+1}/{len(chunks)} complete. Total results: {len(all_results)}")

    return all_results


def load_state(run_dir: Path) -> Dict[str, Any]:
    """Load saved state from a previous run."""
    state_file = run_dir / "state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}


def save_state(run_dir: Path, state: Dict[str, Any]):
    """Save current state for resumption."""
    state_file = run_dir / "state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    logger.info(f"State saved to {state_file}")


async def run_memory_system_batch_benchmark(
    variant: str = "long",
    sample_size: int = None,
    output_dir: Path = None,
    enable_prefilter: bool = True,
    resume_dir: Path = None
):
    """Run HaluMem benchmark using MemorySystem + Batch API.

    Args:
        variant: "long" or "medium" benchmark variant
        sample_size: Limit QA pairs for testing
        output_dir: Base directory for results
        enable_prefilter: Use embedding prefiltering
        resume_dir: Path to previous run to resume from
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    client = openai.AsyncOpenAI(api_key=api_key)
    batch_processor = BatchProcessor(client)

    # Real-time model for scoring (cheap, no batching needed)
    # Note: gpt-5-nano only supports temperature=1 (default)
    async def gpt5_nano(prompt: str) -> str:
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def embed_fn(texts: List[str]) -> List[List[float]]:
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [item.embedding for item in response.data]

    # Setup - handle resume or new run
    output_dir = output_dir or Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume_dir:
        run_dir = Path(resume_dir)
        if not run_dir.exists():
            raise ValueError(f"Resume directory not found: {resume_dir}")
        state = load_state(run_dir)
        if not state:
            raise ValueError(f"No state.json found in {resume_dir}")
        run_id = state["run_id"]
        variant = state.get("variant", variant)
        sample_size = state.get("sample_size", sample_size)
        enable_prefilter = state.get("prefilter_enabled", enable_prefilter)
        logger.info(f"Resuming run: {run_id}")
        logger.info(f"Current phase: {state.get('current_phase', 'unknown')}")
    else:
        run_id = f"memsys_batch_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "run_id": run_id,
            "variant": variant,
            "sample_size": sample_size,
            "prefilter_enabled": enable_prefilter,
            "current_phase": "init",
            "phases_completed": []
        }
        save_state(run_dir, state)
        logger.info(f"Starting new run: {run_id}")

    config = HaluMemConfig.for_long() if variant == "long" else HaluMemConfig.for_medium()
    if sample_size:
        config.sample_size = sample_size

    # Initialize MemorySystem
    db_path = str(run_dir / "memories.db")
    initial_top_k = 1000

    memory_system = MemorySystem(
        small_model_fn=gpt5_nano,
        db_path=db_path,
        relevance_threshold=0.5,
        max_memories=20,
        batch_size=10,
        embedding_fn=embed_fn if enable_prefilter else None,
        enable_prefilter=enable_prefilter,
        prefilter_top_k=initial_top_k
    )

    # Load dataset
    dataset = HaluMemDataset(config)
    await dataset.load()
    stats = dataset.get_stats()
    logger.info(f"Dataset loaded: {stats}")

    # Load or initialize results
    results_file = run_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
    else:
        results = {
            "run_id": run_id,
            "variant": variant,
            "sample_size": sample_size,
            "prefilter_enabled": enable_prefilter,
            "started_at": datetime.now().isoformat(),
            "dataset_stats": stats,
            "system": "MemorySystem + Batch API"
        }

    total_start = time.time()
    ground_truth = dataset.get_memories()

    # =========================================================================
    # PHASE 1: EXTRACTION (Batch API)
    # =========================================================================
    if "extraction" in state.get("phases_completed", []):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: EXTRACTION - SKIPPED (already completed)")
        logger.info("=" * 60)

        # Check if memories already exist in the database (from previous run)
        existing_count = memory_system.count_memories()
        if existing_count > 0:
            logger.info(f"Memory store already has {existing_count} memories - skipping re-add")
        else:
            # Load existing extraction results to rebuild memory store
            with open(run_dir / "extraction_raw.json") as f:
                extraction_results = json.load(f)
            for dialogue_id, memories in extraction_results.items():
                idx = int(dialogue_id.split("_")[1])
                for mem in memories:
                    memory_system.add_memory(mem, metadata={"source": "extraction", "dialogue": idx})

        if enable_prefilter:
            mem_count = memory_system.count_memories()
            top_k = max(1000, int(mem_count * 0.1))
            memory_system.update_prefilter_config(top_k=top_k)
            # Only generate embeddings if there are memories without them
            without_embeddings = memory_system.storage.count_memories_without_embeddings()
            if without_embeddings > 0:
                logger.info(f"Generating embeddings for {without_embeddings} memories")
                await memory_system.generate_embeddings()
            else:
                logger.info("All memories already have embeddings - skipping embedding generation")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: EXTRACTION (Batch API with chunking)")
        logger.info("=" * 60)

        start_time = time.time()
        dialogues = dataset.get_dialogues()

        # Build all extraction requests
        extraction_requests = []
        for i, dialogue in enumerate(dialogues):
            dialogue_text = "\n".join([
                f"{turn.role}: {turn.content}"
                for turn in dialogue
            ])

            prompt = f"""Extract important facts, preferences, and personal information from this conversation.
Return each memory as a separate line. Be concise - each memory should be one clear statement.
Focus on facts about the user that would be useful to remember for future conversations.

CONVERSATION:
{dialogue_text}

MEMORIES (one per line):"""

            extraction_requests.append(batch_processor._create_batch_request(
                custom_id=f"extract_{i}",
                model="gpt-5.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            ))

        # Chunk requests to fit under token limit
        extraction_chunks = chunk_requests(extraction_requests)
        logger.info(f"Processing {len(extraction_requests)} extraction requests in {len(extraction_chunks)} chunks")

        # Process chunks with resume support
        state["current_phase"] = "extraction"
        save_state(run_dir, state)

        extraction_responses = await process_chunked_batch(
            batch_processor=batch_processor,
            chunks=extraction_chunks,
            description="HaluMem extraction - MemorySystem",
            state=state,
            state_key="extraction_chunks",
            run_dir=run_dir
        )

        # Parse and store extracted memories
        extraction_results = {}
        for custom_id, response in extraction_responses.items():
            idx = int(custom_id.split("_")[1])
            memories = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line.startswith(('-', '*', '•')):
                    line = line[1:].strip()
                elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
                    line = line[2:].strip()
                if line and not line.startswith("ERROR"):
                    memories.append(line)
                    # Store in MemorySystem
                    memory_system.add_memory(line, metadata={"source": "extraction", "dialogue": idx})
            extraction_results[f"dialogue_{idx}"] = memories

        total_extracted = sum(len(m) for m in extraction_results.values())
        logger.info(f"Extracted {total_extracted} memories")

        # Save extraction results FIRST (before embeddings, so we can resume)
        with open(run_dir / "extraction_raw.json", 'w') as f:
            json.dump(extraction_results, f, indent=2)
        logger.info(f"Saved extraction results to {run_dir / 'extraction_raw.json'}")

        # Generate embeddings with progress tracking
        if enable_prefilter:
            mem_count = memory_system.count_memories()
            top_k = max(1000, int(mem_count * 0.1))
            memory_system.update_prefilter_config(top_k=top_k)
            logger.info(f"Prefilter top_k: {top_k}")

            # Check how many already have embeddings
            stats = memory_system.get_stats()
            already_embedded = stats.get("memories_with_embeddings", 0)
            need_embedding = stats.get("memories_without_embeddings", mem_count)
            logger.info(f"Embeddings: {already_embedded} done, {need_embedding} remaining")

            if need_embedding > 0:
                logger.info(f"Generating embeddings for {need_embedding} memories...")
                # Generate with progress logging
                batch_size = 100
                generated = await memory_system.generate_embeddings(batch_size=batch_size)
                logger.info(f"Generated {generated} embeddings")
            else:
                logger.info("All memories already have embeddings")

        # Compute metrics
        all_extracted = []
        for mems in extraction_results.values():
            all_extracted.extend(mems)

        weights = {m.memory_id: m.importance for m in ground_truth.values()}
        matched_gt_ids = set()
        for ext_mem in all_extracted:
            ext_lower = ext_mem.lower()
            for gt_id, gt_mem in ground_truth.items():
                if ext_lower in gt_mem.text.lower() or gt_mem.text.lower() in ext_lower:
                    matched_gt_ids.add(gt_id)

        extraction_metrics = compute_extraction_metrics(
            extracted_memories=list(matched_gt_ids),
            ground_truth_memories=list(ground_truth.keys()),
            importance_weights=weights,
            target_memory_ids={m.memory_id for m in ground_truth.values() if m.is_target}
        )

        results["extraction"] = {
            "metrics": extraction_metrics.to_dict(),
            "time_seconds": time.time() - start_time,
            "total_extracted": total_extracted,
            "matched": len(matched_gt_ids),
            "num_chunks": len(extraction_chunks)
        }
        logger.info(f"Extraction: Recall={extraction_metrics.recall:.2%}")

        # Mark phase complete
        state["phases_completed"].append("extraction")
        state["current_phase"] = "extraction_complete"
        save_state(run_dir, state)

        # Save results incrementally
        with open(run_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

    # =========================================================================
    # PHASE 2: UPDATING (Batched retrieval + Batch decisions)
    # =========================================================================
    if "updating" in state.get("phases_completed", []):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: UPDATING - SKIPPED (already completed)")
        logger.info("=" * 60)
        # Load and apply update results to memory store
        with open(run_dir / "updating_raw.json") as f:
            update_results = json.load(f)
        # Note: We'd need scenario_related to properly apply updates on resume
        # For now, the memory store state is persisted in SQLite
    else:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: UPDATING (Batch API for decisions)")
        logger.info("=" * 60)

        start_time = time.time()
        scenarios = dataset.get_update_scenarios()

        # Check if we have retrieval results saved
        retrieval_file = run_dir / "update_retrieval.json"
        if retrieval_file.exists():
            logger.info("Loading saved retrieval results...")
            with open(retrieval_file) as f:
                scenario_related_raw = json.load(f)
            # Reconstruct with memory objects
            scenario_related = {}
            for sid, memories in scenario_related_raw.items():
                # For resume, we just need the text - simplified
                scenario_related[sid] = memories
        else:
            # Step 1: Find related memories for each scenario using BATCHED gpt-5-nano
            logger.info("Finding related memories for each scenario (batched)...")

            # Get memory count for prefilter top_k calculation
            mem_count = memory_system.count_memories()
            logger.info(f"Total memories in store: {mem_count}")

            # Build query list for all scenarios
            scenario_queries = [
                {"query_id": scenario.scenario_id, "query": scenario.new_information}
                for scenario in scenarios
            ]

            # Step 1a: Embedding prefilter to get top candidates per scenario
            prefilter_top_k = max(100, int(mem_count * 0.1))  # top 10% or 100
            logger.info(f"Prefiltering to top {prefilter_top_k} candidates per scenario...")

            prefilter_candidates = await bulk_embedding_prefilter(
                embed_fn=embed_fn,
                queries=scenario_queries,
                all_memories=None,  # Use memory_system instead
                top_k=prefilter_top_k,
                memory_system=memory_system
            )

            # Step 1b: Build queries_and_memories for batch scoring
            queries_and_memories = []
            for scenario in scenarios:
                candidates = prefilter_candidates.get(scenario.scenario_id, [])
                if candidates:
                    queries_and_memories.append({
                        "query_id": scenario.scenario_id,
                        "query": scenario.new_information,
                        "memories": candidates
                    })

            logger.info(f"Batch scoring {len(queries_and_memories)} scenarios with candidates...")

            # Step 1c: Batch score all candidates with gpt-5-nano
            scenario_related = await batch_score_memories(
                batch_processor=batch_processor,
                queries_and_memories=queries_and_memories,
                state=state,
                state_key="update_scoring_chunks",
                run_dir=run_dir,
                threshold=0.5,
                max_per_query=20
            )

            # Save retrieval results
            with open(retrieval_file, 'w') as f:
                json.dump(scenario_related, f, indent=2)
            state["update_retrieval_done"] = True
            save_state(run_dir, state)

        # Step 2: Build update decision prompts
        update_requests = []
        for scenario in scenarios:
            related = scenario_related.get(scenario.scenario_id, [])
            if not related:
                continue

            existing_memory = related[0]["text"] if isinstance(related[0], dict) else related[0].text

            prompt = f"""You are a memory management system. Given an existing memory and new information, decide what action to take.

EXISTING MEMORY:
{existing_memory}

NEW INFORMATION:
{scenario.new_information}

Decide:
- UPDATE: New info contradicts/refines the existing memory
- ADD: New info is additional, doesn't replace existing
- DELETE: Existing memory is now obsolete/wrong
- NOOP: No change needed (already captured or irrelevant)

Respond in exactly this format:
ACTION: <UPDATE|ADD|DELETE|NOOP>
RESULT: <new memory text if UPDATE/ADD, otherwise N/A>
REASONING: <brief explanation>"""

            update_requests.append(batch_processor._create_batch_request(
                custom_id=f"update_{scenario.scenario_id}",
                model="gpt-5.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            ))

        if update_requests:
            # Chunk requests to fit under token limit
            update_chunks = chunk_requests(update_requests)
            logger.info(f"Processing {len(update_requests)} update requests in {len(update_chunks)} chunks")

            state["current_phase"] = "updating"
            save_state(run_dir, state)

            update_responses = await process_chunked_batch(
                batch_processor=batch_processor,
                chunks=update_chunks,
                description="HaluMem updates - MemorySystem",
                state=state,
                state_key="update_chunks",
                run_dir=run_dir
            )
        else:
            update_responses = {}

        # Parse decisions and apply to memory store
        update_results = {}
        predictions = []
        ground_truth_actions = []
        action_types = []

        for scenario in scenarios:
            related = scenario_related.get(scenario.scenario_id, [])

            if not related:
                memory_system.add_memory(
                    scenario.new_information,
                    metadata={"source": "update"}
                )
                predicted = "ADD"
                update_results[scenario.scenario_id] = {"action": "ADD", "reason": "No related memories"}
            else:
                response = update_responses.get(f"update_{scenario.scenario_id}", "")
                decision = {"action": "NOOP", "result": "", "reasoning": ""}

                for line in response.strip().split('\n'):
                    if line.startswith("ACTION:"):
                        action = line.split(":", 1)[1].strip().upper()
                        if action in ("UPDATE", "ADD", "DELETE", "NOOP"):
                            decision["action"] = action
                    elif line.startswith("RESULT:"):
                        decision["result"] = line.split(":", 1)[1].strip()
                    elif line.startswith("REASONING:"):
                        decision["reasoning"] = line.split(":", 1)[1].strip()

                # Apply decision
                mem_id = related[0]["memory_id"] if isinstance(related[0], dict) else related[0].memory_id
                if decision["action"] == "UPDATE" and decision["result"]:
                    memory_system.update_memory(mem_id, text=decision["result"])
                elif decision["action"] == "ADD" and decision["result"]:
                    memory_system.add_memory(decision["result"], metadata={"source": "update"})
                elif decision["action"] == "DELETE":
                    memory_system.delete_memory(mem_id)

                predicted = decision["action"]
                update_results[scenario.scenario_id] = decision

            predictions.append(predicted)
            ground_truth_actions.append(scenario.expected_action)
            action_types.append(scenario.conflict_type)

        with open(run_dir / "updating_raw.json", 'w') as f:
            json.dump(update_results, f, indent=2)

        update_metrics = compute_update_metrics(
            predictions=predictions,
            ground_truth=ground_truth_actions,
            action_types=action_types
        )

        results["updating"] = {
            "metrics": update_metrics.to_dict(),
            "time_seconds": time.time() - start_time,
            "total_scenarios": len(scenarios),
            "num_chunks": len(update_chunks) if update_requests else 0
        }
        logger.info(f"Updating: Accuracy={update_metrics.accuracy:.2%}")

        # Mark phase complete
        state["phases_completed"].append("updating")
        state["current_phase"] = "updating_complete"
        save_state(run_dir, state)

        with open(run_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

    # =========================================================================
    # PHASE 3: QA (Batched retrieval + Batch answers + Batch judging)
    # =========================================================================
    if "qa" in state.get("phases_completed", []):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: QA - SKIPPED (already completed)")
        logger.info("=" * 60)
    else:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: QA (Batch API for answers and judging)")
        logger.info("=" * 60)

        start_time = time.time()
        qa_pairs = dataset.get_qa_pairs(limit=sample_size)

        # Check for saved retrieval results
        qa_retrieval_file = run_dir / "qa_retrieval.json"
        if qa_retrieval_file.exists():
            logger.info("Loading saved QA retrieval results...")
            with open(qa_retrieval_file) as f:
                qa_memories_raw = json.load(f)
            qa_memories = {k: v for k, v in qa_memories_raw.items()}
        else:
            # Step 1: Retrieve relevant memories for each question using BATCHED gpt-5-nano
            logger.info("Retrieving relevant memories for each question (batched)...")

            # Get memory count for prefilter top_k calculation
            mem_count = memory_system.count_memories()
            logger.info(f"Total memories in store: {mem_count}")

            # Build query list for all QA pairs
            qa_queries = [
                {"query_id": qa.qa_id, "query": qa.question}
                for qa in qa_pairs
            ]

            # Step 1a: Embedding prefilter to get top candidates per question
            prefilter_top_k = max(100, int(mem_count * 0.1))  # top 10% or 100
            logger.info(f"Prefiltering to top {prefilter_top_k} candidates per question...")

            prefilter_candidates = await bulk_embedding_prefilter(
                embed_fn=embed_fn,
                queries=qa_queries,
                all_memories=None,  # Use memory_system instead
                top_k=prefilter_top_k,
                memory_system=memory_system
            )

            # Step 1b: Build queries_and_memories for batch scoring
            queries_and_memories = []
            for qa in qa_pairs:
                candidates = prefilter_candidates.get(qa.qa_id, [])
                if candidates:
                    queries_and_memories.append({
                        "query_id": qa.qa_id,
                        "query": qa.question,
                        "memories": candidates
                    })

            logger.info(f"Batch scoring {len(queries_and_memories)} questions with candidates...")

            # Step 1c: Batch score all candidates with gpt-5-nano
            qa_memories = await batch_score_memories(
                batch_processor=batch_processor,
                queries_and_memories=queries_and_memories,
                state=state,
                state_key="qa_scoring_chunks",
                run_dir=run_dir,
                threshold=0.5,
                max_per_query=20
            )

            with open(qa_retrieval_file, 'w') as f:
                json.dump(qa_memories, f, indent=2)
            state["qa_retrieval_done"] = True
            save_state(run_dir, state)

        # Check for saved answer responses
        answer_responses_file = run_dir / "qa_answers.json"

        if answer_responses_file.exists():
            logger.info("Loading saved answer responses...")
            with open(answer_responses_file) as f:
                answer_responses = json.load(f)
        else:
            # Step 2: Build answer generation prompts
            answer_requests = []
            for qa in qa_pairs:
                memories = qa_memories.get(qa.qa_id, [])
                if memories:
                    memory_texts = [m["text"] if isinstance(m, dict) else m.text for m in memories]
                    memory_context = "\n".join([f"- {t}" for t in memory_texts])
                else:
                    memory_context = "(No relevant memories found)"

                prompt = f"""Based on the following memories about the user, answer the question.

MEMORIES:
{memory_context}

QUESTION: {qa.question}

ANSWER:"""

                answer_requests.append(batch_processor._create_batch_request(
                    custom_id=f"answer_{qa.qa_id}",
                    model="gpt-5.1",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                ))

            # Chunk answer requests
            answer_chunks = chunk_requests(answer_requests)
            logger.info(f"Processing {len(answer_requests)} answer requests in {len(answer_chunks)} chunks")

            state["current_phase"] = "qa_answers"
            save_state(run_dir, state)

            answer_responses = await process_chunked_batch(
                batch_processor=batch_processor,
                chunks=answer_chunks,
                description="HaluMem QA answers - MemorySystem",
                state=state,
                state_key="qa_answer_chunks",
                run_dir=run_dir
            )

            # Save complete answer responses
            with open(answer_responses_file, 'w') as f:
                json.dump(answer_responses, f, indent=2)

        # Step 3: Build judging prompts
        judge_requests = []
        for qa in qa_pairs:
            answer = answer_responses.get(f"answer_{qa.qa_id}", "[No answer]")

            prompt = f"""You are evaluating an AI's answer to a question based on stored memories.

QUESTION: {qa.question}

GROUND TRUTH ANSWER: {qa.ground_truth_answer}

AI'S ANSWER: {answer}

Evaluate:
1. Correctness (0.0 to 1.0): Is it factually correct?
2. Hallucination: Does it contain made-up information?
3. Omission: Does it miss important information?

Respond in exactly this format:
CORRECTNESS: <0.0 to 1.0>
HALLUCINATION: <YES|NO>
OMISSION: <YES|NO>
REASONING: <brief explanation>"""

            judge_requests.append(batch_processor._create_batch_request(
                custom_id=f"judge_{qa.qa_id}",
                model="gpt-5.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            ))

        # Chunk judge requests
        judge_chunks = chunk_requests(judge_requests)
        logger.info(f"Processing {len(judge_requests)} judge requests in {len(judge_chunks)} chunks")

        state["current_phase"] = "qa_judging"
        save_state(run_dir, state)

        judge_responses = await process_chunked_batch(
            batch_processor=batch_processor,
            chunks=judge_chunks,
            description="HaluMem QA judging - MemorySystem",
            state=state,
            state_key="qa_judge_chunks",
            run_dir=run_dir
        )

        # Parse results
        qa_results = {}
        correctness_scores = []
        hallucination_flags = []
        omission_flags = []
        question_types = []

        for qa in qa_pairs:
            answer = answer_responses.get(f"answer_{qa.qa_id}", "[Error]")
            judgment = judge_responses.get(f"judge_{qa.qa_id}", "")

            correctness = 0.0
            hallucination = False
            omission = False
            reasoning = ""

            for line in judgment.strip().split('\n'):
                if line.startswith("CORRECTNESS:"):
                    try:
                        correctness = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("HALLUCINATION:"):
                    hallucination = "YES" in line.upper()
                elif line.startswith("OMISSION:"):
                    omission = "YES" in line.upper()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            memories_for_qa = qa_memories.get(qa.qa_id, [])
            qa_results[qa.qa_id] = {
                "question": qa.question,
                "answer": answer,
                "ground_truth": qa.ground_truth_answer,
                "memories_retrieved": len(memories_for_qa),
                "correctness": correctness,
                "hallucination": hallucination,
                "omission": omission,
                "reasoning": reasoning
            }

            correctness_scores.append(correctness)
            hallucination_flags.append(hallucination)
            omission_flags.append(omission)
            question_types.append(qa.question_type)

        with open(run_dir / "qa_raw.json", 'w') as f:
            json.dump(qa_results, f, indent=2)

        qa_metrics = compute_qa_metrics(
            correctness_scores=correctness_scores,
            hallucination_flags=hallucination_flags,
            omission_flags=omission_flags,
            question_types=question_types
        )

        results["qa"] = {
            "metrics": qa_metrics.to_dict(),
            "time_seconds": time.time() - start_time,
            "total_qa_pairs": len(qa_pairs),
            "answer_chunks": len(answer_chunks) if 'answer_chunks' in dir() else 0,
            "judge_chunks": len(judge_chunks)
        }
        logger.info(f"QA: Correctness={qa_metrics.correctness:.2%}")

        # Mark phase complete
        state["phases_completed"].append("qa")
        state["current_phase"] = "complete"
        save_state(run_dir, state)

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    results["total_time_seconds"] = time.time() - total_start
    results["completed_at"] = datetime.now().isoformat()
    results["final_stats"] = memory_system.get_stats()

    with open(run_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("HALUMEM BENCHMARK - MEMORYSYSTEM + BATCH API")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Variant: {variant}")
    print(f"Prefilter: {'enabled' if enable_prefilter else 'disabled'}")
    print(f"Output: {run_dir}")
    print()

    print("EXTRACTION:")
    m = results["extraction"]["metrics"]
    print(f"  Recall: {m.get('recall', 0):.2%}")
    print(f"  Weighted Recall: {m.get('weighted_recall', 0):.2%}")
    print(f"  Precision: {m.get('precision', 0):.2%}")
    print()

    print("UPDATING:")
    m = results["updating"]["metrics"]
    print(f"  Accuracy: {m.get('accuracy', 0):.2%}")
    print(f"  Omission Rate: {m.get('omission_rate', 0):.2%}")
    print(f"  Hallucination Rate: {m.get('hallucination_rate', 0):.2%}")
    print()

    print("QA:")
    m = results["qa"]["metrics"]
    print(f"  Correctness: {m.get('correctness', 0):.2%}")
    print(f"  Hallucination Rate: {m.get('hallucination_rate', 0):.2%}")
    print(f"  Omission Rate: {m.get('omission_rate', 0):.2%}")
    print()

    print(f"Total time: {results['total_time_seconds']:.1f}s")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run HaluMem benchmark using MemorySystem + Batch API"
    )
    parser.add_argument("--variant", choices=["long", "medium"], default="long")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--no-prefilter", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to previous run directory to resume from"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    asyncio.run(run_memory_system_batch_benchmark(
        variant=args.variant,
        sample_size=args.sample_size,
        enable_prefilter=not args.no_prefilter,
        resume_dir=Path(args.resume) if args.resume else None
    ))


if __name__ == "__main__":
    main()
