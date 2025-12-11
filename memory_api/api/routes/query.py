"""
Query and retrieval API routes.

These routes integrate with the core memory system to provide:
- Semantic memory retrieval
- Question answering with memories
- Memory extraction from conversations

This is where the API layer connects to the core memory science.
"""

import time
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import Memory, Organization
from api.middleware.auth import CurrentOrg, CurrentAPIKey, require_scope
from api.middleware.rate_limit import check_rate_limit
from api.middleware.usage_tracking import track_usage
from api.models.requests import QueryRequest, QueryWithAnswerRequest, ExtractMemoriesRequest
from api.models.responses import (
    QueryResponse,
    QueryResponseData,
    QueryWithAnswerResponse,
    QueryWithAnswerData,
    ScoredMemoryData,
    SuccessResponse,
    ResponseMeta,
    UsageMeta,
)
from config.settings import settings


# =============================================================================
# Core Memory System Integration
# =============================================================================
# The memory_lib is the "science" - it contains the core memory algorithms.
# We import and use it here, keeping it abstracted from the API boilerplate.

# Add memory_lib to path
MEMORY_LIB_PATH = Path(__file__).parent.parent.parent.parent / "memory_lib"
if str(MEMORY_LIB_PATH) not in sys.path:
    sys.path.insert(0, str(MEMORY_LIB_PATH))

# Import core memory system (lazy load to avoid import errors if not available)
_memory_system_cache = {}


async def get_memory_system_for_org(org_id: str, db: AsyncSession):
    """
    Get or create a memory system instance for an organization.

    The memory system is the core science - this function adapts it
    to work with our multi-tenant API database.
    """
    # For MVP, we'll implement a simplified adapter that uses our PostgreSQL
    # storage instead of the memory_lib's SQLite storage.
    # This keeps the API working while allowing upgrade to full memory_lib later.

    if org_id not in _memory_system_cache:
        _memory_system_cache[org_id] = APIMemoryAdapter(org_id, db)

    return _memory_system_cache[org_id]


class APIMemoryAdapter:
    """
    Adapter that provides memory_lib-like interface using API database.

    This bridges the gap between the API's PostgreSQL storage and
    the core memory system algorithms. For the MVP, we implement
    simplified versions of the core algorithms here. As the system
    matures, this can delegate to the full memory_lib.
    """

    def __init__(self, org_id: str, db: AsyncSession):
        self.org_id = org_id
        self.db = db

    async def get_all_memories(self) -> list[dict]:
        """Get all memories for the organization."""
        result = await self.db.execute(
            select(Memory).where(
                Memory.org_id == self.org_id,
                Memory.is_active == True,
            )
        )
        memories = result.scalars().all()
        return [
            {
                "id": m.id,
                "text": m.text,
                "metadata": m.metadata_,
                "embedding": m.embedding,
            }
            for m in memories
        ]

    async def retrieve_relevant_memories(
        self,
        context: str,
        max_memories: int = 20,
        relevance_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Retrieve memories relevant to the given context.

        For MVP, this uses a simplified keyword matching approach.
        The full implementation would use the memory_lib's LLM-based
        relevance scoring with embedding prefiltering.
        """
        memories = await self.get_all_memories()

        if not memories:
            return []

        # MVP: Simple keyword relevance scoring
        # TODO: Replace with memory_lib's LLM-based scoring
        context_lower = context.lower()
        context_words = set(context_lower.split())

        scored_memories = []
        for memory in memories:
            text_lower = memory["text"].lower()
            text_words = set(text_lower.split())

            # Calculate simple relevance score
            common_words = context_words & text_words
            if common_words:
                # Jaccard similarity
                score = len(common_words) / len(context_words | text_words)
                # Boost for exact substring matches
                if any(word in text_lower for word in context_words if len(word) > 3):
                    score = min(1.0, score + 0.3)
            else:
                score = 0.0

            if score >= relevance_threshold:
                scored_memories.append({
                    **memory,
                    "relevance_score": score,
                    "relevance_reasoning": f"Matched keywords: {', '.join(common_words)}" if common_words else None,
                })

        # Sort by relevance and limit
        scored_memories.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_memories[:max_memories]


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/query", tags=["Query & Retrieval"])


def build_response_meta(request: Request, usage: Optional[UsageMeta] = None) -> ResponseMeta:
    """Build standard response metadata."""
    start_time = getattr(request.state, "start_time", time.time())
    return ResponseMeta(
        request_id=getattr(request.state, "request_id", "unknown"),
        processing_time_ms=int((time.time() - start_time) * 1000),
        usage=usage,
    )


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query for relevant memories",
    description="Find memories relevant to a given context without generating an answer.",
)
async def query_memories(
    request: Request,
    body: QueryRequest,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Query for relevant memories based on context."""
    # Get memory system adapter
    memory_system = await get_memory_system_for_org(org.id, db)

    # Get total memory count
    all_memories = await memory_system.get_all_memories()
    total_count = len(all_memories)

    # Retrieve relevant memories
    relevant_memories = await memory_system.retrieve_relevant_memories(
        context=body.context,
        max_memories=body.max_memories or 20,
        relevance_threshold=body.relevance_threshold or 0.5,
    )

    # Build response
    scored_memories = [
        ScoredMemoryData(
            id=m["id"],
            text=m["text"],
            relevance_score=m["relevance_score"],
            relevance_reasoning=m.get("relevance_reasoning"),
            metadata=m.get("metadata") if body.include_metadata else None,
        )
        for m in relevant_memories
    ]

    # Track usage
    await track_usage(
        request=request,
        response=None,
        memories_read=len(relevant_memories),
    )

    return QueryResponse(
        data=QueryResponseData(
            memories=scored_memories,
            query_context=body.context,
            total_memories_searched=total_count,
        ),
        meta=build_response_meta(
            request,
            UsageMeta(memories_accessed=len(relevant_memories)),
        ),
    )


@router.post(
    "/answer",
    response_model=QueryWithAnswerResponse,
    summary="Query and generate answer",
    description="Find relevant memories and generate an answer using an LLM.",
)
async def query_with_answer(
    request: Request,
    body: QueryWithAnswerRequest,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Query memories and generate an answer."""
    # Get memory system adapter
    memory_system = await get_memory_system_for_org(org.id, db)

    # Retrieve relevant memories
    relevant_memories = await memory_system.retrieve_relevant_memories(
        context=body.context,
        max_memories=body.max_memories or 20,
        relevance_threshold=body.relevance_threshold or 0.5,
    )

    # Build prompt for answer generation
    memory_context = "\n".join([
        f"- {m['text']}"
        for m in relevant_memories
    ])

    if body.prompt_template:
        prompt = body.prompt_template.format(
            context=body.context,
            memories=memory_context,
        )
    else:
        prompt = f"""Based on the following memories about the user, answer the question.

Memories:
{memory_context}

Question: {body.context}

Answer based only on the memories provided. If the memories don't contain relevant information, say so."""

    # Generate answer using OpenAI
    # For MVP, we'll use a simple completion
    model = body.model or settings.openai_model

    try:
        import openai
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided memories."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content
        tokens_input = response.usage.prompt_tokens
        tokens_output = response.usage.completion_tokens

    except Exception as e:
        # If OpenAI fails, return a simple response
        if relevant_memories:
            answer = f"Based on the memories, here's what I found:\n\n" + "\n".join([
                f"- {m['text']}" for m in relevant_memories[:5]
            ])
        else:
            answer = "No relevant memories found for this query."
        tokens_input = 0
        tokens_output = 0

    # Build response
    scored_memories = [
        ScoredMemoryData(
            id=m["id"],
            text=m["text"],
            relevance_score=m["relevance_score"],
            relevance_reasoning=m.get("relevance_reasoning"),
            metadata=m.get("metadata"),
        )
        for m in relevant_memories
    ]

    # Track usage
    await track_usage(
        request=request,
        response=None,
        memories_read=len(relevant_memories),
        tokens_input=tokens_input,
        tokens_output=tokens_output,
    )

    return QueryWithAnswerResponse(
        data=QueryWithAnswerData(
            answer=answer,
            memories_used=scored_memories,
            confidence=None,  # Could implement confidence scoring
            model=model,
        ),
        meta=build_response_meta(
            request,
            UsageMeta(
                memories_accessed=len(relevant_memories),
                tokens_processed=tokens_input + tokens_output,
                model_calls=1,
            ),
        ),
    )


@router.post(
    "/extract",
    response_model=SuccessResponse,
    summary="Extract memories from content",
    description="Extract and store memories from a conversation or document.",
)
async def extract_memories(
    request: Request,
    body: ExtractMemoriesRequest,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["write"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Extract memories from content using an LLM."""
    # Build extraction prompt
    prompt = f"""Extract key facts, preferences, and information from the following {body.content_type}.
Return each memory as a separate line starting with "- ".
Focus on information that would be useful to remember for future interactions.

Content:
{body.content}

Extracted memories:"""

    try:
        import openai
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You extract key information as memorable facts."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        extracted_text = response.choices[0].message.content
        tokens_input = response.usage.prompt_tokens
        tokens_output = response.usage.completion_tokens

        # Parse extracted memories
        lines = extracted_text.strip().split("\n")
        memories_to_create = []
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                memory_text = line[2:].strip()
                if memory_text:
                    memories_to_create.append(memory_text)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "EXTRACTION_FAILED",
                "message": f"Failed to extract memories: {str(e)}",
            },
        )

    # Create memories
    created_ids = []
    for text in memories_to_create:
        memory = Memory(
            org_id=org.id,
            text=text,
            metadata_={
                **(body.metadata or {}),
                "source": "extraction",
                "content_type": body.content_type,
            },
        )
        db.add(memory)
        await db.flush()
        created_ids.append(memory.id)

    # Track usage
    await track_usage(
        request=request,
        response=None,
        memories_written=len(created_ids),
        tokens_input=tokens_input,
        tokens_output=tokens_output,
    )

    return SuccessResponse(
        data={
            "extracted_count": len(created_ids),
            "memory_ids": created_ids,
        },
        meta=build_response_meta(
            request,
            UsageMeta(
                memories_accessed=len(created_ids),
                tokens_processed=tokens_input + tokens_output,
                model_calls=1,
            ),
        ),
    )
