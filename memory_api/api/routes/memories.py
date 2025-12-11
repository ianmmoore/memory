"""
Memory CRUD API routes.

Endpoints:
- POST   /memories          Create a memory
- GET    /memories          List memories
- GET    /memories/{id}     Get single memory
- PUT    /memories/{id}     Update memory
- DELETE /memories/{id}     Delete memory
- DELETE /memories          Delete all memories
"""

import time
from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import Memory, Organization
from api.middleware.auth import CurrentOrg, CurrentAPIKey, require_scope
from api.middleware.rate_limit import check_rate_limit
from api.middleware.usage_tracking import track_usage
from api.models.requests import CreateMemoryRequest, UpdateMemoryRequest
from api.models.responses import (
    MemoryResponse,
    MemoryListResponse,
    MemoryData,
    SuccessResponse,
    ResponseMeta,
    UsageMeta,
    PaginationMeta,
)

router = APIRouter(prefix="/memories", tags=["Memories"])


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
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a memory",
    description="Create a new memory with optional metadata.",
)
async def create_memory(
    request: Request,
    body: CreateMemoryRequest,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["write"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Create a new memory."""
    # Check if custom ID already exists
    if body.memory_id:
        existing = await db.execute(
            select(Memory).where(
                Memory.org_id == org.id,
                Memory.id == body.memory_id,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "MEMORY_ID_EXISTS",
                    "message": f"Memory with ID '{body.memory_id}' already exists",
                },
            )

    # Create memory
    memory = Memory(
        org_id=org.id,
        text=body.text,
        metadata_=body.metadata or {},
    )

    if body.memory_id:
        memory.id = body.memory_id

    db.add(memory)
    await db.flush()
    await db.refresh(memory)

    # Track usage
    await track_usage(
        request=request,
        response=None,
        memories_written=1,
    )

    return MemoryResponse(
        data=MemoryData(
            id=memory.id,
            text=memory.text,
            metadata=memory.metadata_,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        ),
        meta=build_response_meta(request, UsageMeta(memories_accessed=1)),
    )


@router.get(
    "",
    response_model=MemoryListResponse,
    summary="List memories",
    description="List all memories with pagination.",
)
async def list_memories(
    request: Request,
    page: int = Query(default=1, ge=1, description="Page number"),
    per_page: int = Query(default=50, ge=1, le=100, description="Items per page"),
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """List memories with pagination."""
    # Get total count
    count_result = await db.execute(
        select(func.count(Memory.id)).where(
            Memory.org_id == org.id,
            Memory.is_active == True,
        )
    )
    total = count_result.scalar_one()

    # Get page of memories
    offset = (page - 1) * per_page
    result = await db.execute(
        select(Memory)
        .where(Memory.org_id == org.id, Memory.is_active == True)
        .order_by(Memory.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    memories = result.scalars().all()

    total_pages = (total + per_page - 1) // per_page

    return MemoryListResponse(
        data=[
            MemoryData(
                id=m.id,
                text=m.text,
                metadata=m.metadata_,
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in memories
        ],
        pagination=PaginationMeta(
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        ),
        meta=build_response_meta(request, UsageMeta(memories_accessed=len(memories))),
    )


@router.get(
    "/{memory_id}",
    response_model=MemoryResponse,
    summary="Get a memory",
    description="Get a single memory by ID.",
)
async def get_memory(
    request: Request,
    memory_id: str,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Get a single memory."""
    result = await db.execute(
        select(Memory).where(
            Memory.org_id == org.id,
            Memory.id == memory_id,
            Memory.is_active == True,
        )
    )
    memory = result.scalar_one_or_none()

    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "MEMORY_NOT_FOUND",
                "message": f"Memory '{memory_id}' not found",
            },
        )

    return MemoryResponse(
        data=MemoryData(
            id=memory.id,
            text=memory.text,
            metadata=memory.metadata_,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        ),
        meta=build_response_meta(request, UsageMeta(memories_accessed=1)),
    )


@router.put(
    "/{memory_id}",
    response_model=MemoryResponse,
    summary="Update a memory",
    description="Update an existing memory's text and/or metadata.",
)
async def update_memory(
    request: Request,
    memory_id: str,
    body: UpdateMemoryRequest,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["write"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Update a memory."""
    result = await db.execute(
        select(Memory).where(
            Memory.org_id == org.id,
            Memory.id == memory_id,
            Memory.is_active == True,
        )
    )
    memory = result.scalar_one_or_none()

    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "MEMORY_NOT_FOUND",
                "message": f"Memory '{memory_id}' not found",
            },
        )

    # Update fields
    if body.text is not None:
        memory.text = body.text
        # Clear embedding since text changed
        memory.embedding = None
        memory.embedding_model = None

    if body.metadata is not None:
        memory.metadata_ = body.metadata

    memory.updated_at = datetime.now(timezone.utc)
    await db.flush()
    await db.refresh(memory)

    # Track usage
    await track_usage(
        request=request,
        response=None,
        memories_written=1,
    )

    return MemoryResponse(
        data=MemoryData(
            id=memory.id,
            text=memory.text,
            metadata=memory.metadata_,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        ),
        meta=build_response_meta(request, UsageMeta(memories_accessed=1)),
    )


@router.delete(
    "/{memory_id}",
    response_model=SuccessResponse,
    summary="Delete a memory",
    description="Delete a memory by ID (soft delete).",
)
async def delete_memory(
    request: Request,
    memory_id: str,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["delete"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Delete a memory (soft delete)."""
    result = await db.execute(
        select(Memory).where(
            Memory.org_id == org.id,
            Memory.id == memory_id,
            Memory.is_active == True,
        )
    )
    memory = result.scalar_one_or_none()

    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "MEMORY_NOT_FOUND",
                "message": f"Memory '{memory_id}' not found",
            },
        )

    # Soft delete
    memory.is_active = False
    memory.updated_at = datetime.now(timezone.utc)

    return SuccessResponse(
        data={"deleted": True, "memory_id": memory_id},
        meta=build_response_meta(request),
    )


@router.delete(
    "",
    response_model=SuccessResponse,
    summary="Delete all memories",
    description="Delete all memories for the organization. Requires confirmation.",
)
async def delete_all_memories(
    request: Request,
    confirm: bool = Query(
        ...,
        description="Must be true to confirm deletion",
    ),
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["delete", "admin"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Delete all memories for the organization."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "CONFIRMATION_REQUIRED",
                "message": "Set confirm=true to delete all memories",
            },
        )

    # Count before deletion
    count_result = await db.execute(
        select(func.count(Memory.id)).where(
            Memory.org_id == org.id,
            Memory.is_active == True,
        )
    )
    count = count_result.scalar_one()

    # Soft delete all
    await db.execute(
        delete(Memory).where(Memory.org_id == org.id)
    )

    return SuccessResponse(
        data={"deleted": True, "count": count},
        meta=build_response_meta(request),
    )


@router.get(
    "/count",
    response_model=SuccessResponse,
    summary="Count memories",
    description="Get the total count of memories.",
)
async def count_memories(
    request: Request,
    org: Organization = Depends(CurrentOrg),
    api_key: CurrentAPIKey = Depends(require_scope(["read"])),
    rate_limit: dict = Depends(check_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """Get memory count."""
    result = await db.execute(
        select(func.count(Memory.id)).where(
            Memory.org_id == org.id,
            Memory.is_active == True,
        )
    )
    count = result.scalar_one()

    return SuccessResponse(
        data={"count": count},
        meta=build_response_meta(request),
    )
