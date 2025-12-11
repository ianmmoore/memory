"""
Pydantic response models for consistent API responses.

All API responses follow a standard format:
{
    "success": true/false,
    "data": { ... } or null,
    "error": { ... } or null,
    "meta": { ... }
}
"""

from datetime import datetime
from typing import Optional, Any, Generic, TypeVar
from pydantic import BaseModel, Field, ConfigDict

T = TypeVar("T")


# =============================================================================
# Meta Objects
# =============================================================================

class PaginationMeta(BaseModel):
    """Pagination metadata for list endpoints."""
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number (1-indexed)")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")


class UsageMeta(BaseModel):
    """Usage metadata included in responses."""
    tokens_processed: Optional[int] = Field(default=None, description="Tokens consumed")
    memories_accessed: Optional[int] = Field(default=None, description="Memories read/written")
    model_calls: Optional[int] = Field(default=None, description="LLM API calls made")


class ResponseMeta(BaseModel):
    """Standard response metadata."""
    request_id: str = Field(..., description="Unique request identifier")
    processing_time_ms: int = Field(..., description="Request processing time in milliseconds")
    usage: Optional[UsageMeta] = Field(default=None, description="Resource usage for this request")


# =============================================================================
# Error Response
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(default=None, description="Additional error details")
    field: Optional[str] = Field(default=None, description="Field that caused the error (for validation)")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    success: bool = Field(default=False)
    data: None = None
    error: ErrorDetail
    meta: Optional[ResponseMeta] = None


# =============================================================================
# Success Response
# =============================================================================

class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = Field(default=True)
    data: Optional[dict[str, Any]] = None
    error: None = None
    meta: Optional[ResponseMeta] = None


# =============================================================================
# Memory Responses
# =============================================================================

class MemoryData(BaseModel):
    """Individual memory data."""
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique memory identifier")
    text: str = Field(..., description="Memory content")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Memory metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class MemoryResponse(BaseModel):
    """Response for single memory operations."""
    success: bool = True
    data: MemoryData
    error: None = None
    meta: Optional[ResponseMeta] = None


class MemoryListResponse(BaseModel):
    """Response for listing memories."""
    success: bool = True
    data: list[MemoryData]
    error: None = None
    meta: Optional[ResponseMeta] = None
    pagination: Optional[PaginationMeta] = None


# =============================================================================
# Query Responses
# =============================================================================

class ScoredMemoryData(BaseModel):
    """Memory with relevance score from query."""
    id: str = Field(..., description="Memory identifier")
    text: str = Field(..., description="Memory content")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    relevance_reasoning: Optional[str] = Field(default=None, description="Why this memory is relevant")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Memory metadata")


class QueryResponseData(BaseModel):
    """Query response data."""
    memories: list[ScoredMemoryData] = Field(..., description="Relevant memories")
    query_context: str = Field(..., description="The original query context")
    total_memories_searched: int = Field(..., description="Total memories in search space")


class QueryResponse(BaseModel):
    """Response for memory query."""
    success: bool = True
    data: QueryResponseData
    error: None = None
    meta: Optional[ResponseMeta] = None


class QueryWithAnswerData(BaseModel):
    """Query with answer response data."""
    answer: str = Field(..., description="Generated answer based on memories")
    memories_used: list[ScoredMemoryData] = Field(..., description="Memories used to generate answer")
    confidence: Optional[float] = Field(default=None, description="Answer confidence (0-1)")
    model: str = Field(..., description="Model used for answer generation")


class QueryWithAnswerResponse(BaseModel):
    """Response for query with answer generation."""
    success: bool = True
    data: QueryWithAnswerData
    error: None = None
    meta: Optional[ResponseMeta] = None


# =============================================================================
# Batch Responses
# =============================================================================

class BatchJobData(BaseModel):
    """Batch job status data."""
    job_id: str = Field(..., description="Unique batch job identifier")
    status: str = Field(..., description="Job status: pending, processing, completed, failed")
    progress: Optional[float] = Field(default=None, description="Progress percentage (0-100)")
    total_items: int = Field(..., description="Total items in batch")
    processed_items: int = Field(default=0, description="Items processed so far")
    failed_items: int = Field(default=0, description="Items that failed")
    created_at: datetime = Field(..., description="Job creation time")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion time")
    result: Optional[dict[str, Any]] = Field(default=None, description="Job result when completed")


class BatchJobResponse(BaseModel):
    """Response for batch operations."""
    success: bool = True
    data: BatchJobData
    error: None = None
    meta: Optional[ResponseMeta] = None


# =============================================================================
# Usage & Billing Responses
# =============================================================================

class UsageData(BaseModel):
    """Current usage data."""
    period: str = Field(..., description="Billing period (YYYY-MM)")
    api_calls: int = Field(..., description="Total API calls")
    tokens_processed: int = Field(..., description="Total tokens (input + output)")
    tokens_input: int = Field(..., description="Input tokens")
    tokens_output: int = Field(..., description="Output tokens")
    memories_stored: int = Field(..., description="Current memory count")
    memories_read: int = Field(..., description="Memories read this period")
    memories_written: int = Field(..., description="Memories written this period")
    embeddings_generated: int = Field(..., description="Embeddings created")
    estimated_cost_cents: int = Field(..., description="Estimated cost in cents")


class UsageLimits(BaseModel):
    """Usage limits for current plan."""
    api_calls: Optional[int] = Field(default=None, description="API call limit (null=unlimited)")
    tokens_processed: Optional[int] = Field(default=None, description="Token limit")
    memories_stored: Optional[int] = Field(default=None, description="Memory storage limit")
    embeddings_generated: Optional[int] = Field(default=None, description="Embedding limit")


class UsageResponseData(BaseModel):
    """Usage response data."""
    current: UsageData
    limits: UsageLimits
    percent_used: dict[str, float] = Field(..., description="Percentage of each limit used")


class UsageResponse(BaseModel):
    """Response for usage endpoint."""
    success: bool = True
    data: UsageResponseData
    error: None = None
    meta: Optional[ResponseMeta] = None


# =============================================================================
# Account Responses
# =============================================================================

class PlanData(BaseModel):
    """Pricing plan data."""
    id: str
    name: str
    description: Optional[str] = None
    base_price_cents: Optional[int] = None
    billing_period: str = "monthly"


class AccountData(BaseModel):
    """Account information."""
    org_id: str = Field(..., description="Organization ID")
    name: str = Field(..., description="Organization name")
    email: str = Field(..., description="Contact email")
    plan: PlanData = Field(..., description="Current plan")
    is_active: bool = Field(..., description="Account active status")
    created_at: datetime = Field(..., description="Account creation date")


class AccountResponse(BaseModel):
    """Response for account endpoint."""
    success: bool = True
    data: AccountData
    error: None = None
    meta: Optional[ResponseMeta] = None


# =============================================================================
# API Key Responses
# =============================================================================

class APIKeyData(BaseModel):
    """API key data (without the full key)."""
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Key ID")
    name: str = Field(..., description="Key name")
    key_prefix: str = Field(..., description="Key prefix for identification")
    environment: str = Field(..., description="Environment (live/test)")
    scopes: list[str] = Field(..., description="Permission scopes")
    is_active: bool = Field(..., description="Key active status")
    created_at: datetime = Field(..., description="Creation date")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration date")
    last_used_at: Optional[datetime] = Field(default=None, description="Last usage date")


class APIKeyResponse(BaseModel):
    """Response for API key operations (existing keys)."""
    success: bool = True
    data: APIKeyData
    error: None = None
    meta: Optional[ResponseMeta] = None


class APIKeyCreatedData(BaseModel):
    """API key data including the full key (only shown once on creation)."""
    id: str = Field(..., description="Key ID")
    name: str = Field(..., description="Key name")
    key: str = Field(..., description="Full API key - SAVE THIS, shown only once!")
    key_prefix: str = Field(..., description="Key prefix for identification")
    environment: str = Field(..., description="Environment (live/test)")
    scopes: list[str] = Field(..., description="Permission scopes")
    created_at: datetime = Field(..., description="Creation date")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration date")


class APIKeyCreatedResponse(BaseModel):
    """Response for newly created API key (includes full key)."""
    success: bool = True
    data: APIKeyCreatedData
    error: None = None
    meta: Optional[ResponseMeta] = None


class APIKeyListResponse(BaseModel):
    """Response for listing API keys."""
    success: bool = True
    data: list[APIKeyData]
    error: None = None
    meta: Optional[ResponseMeta] = None


# =============================================================================
# Health Check Response
# =============================================================================

class HealthData(BaseModel):
    """Health check data."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    database: str = Field(..., description="Database connection status")
    redis: str = Field(..., description="Redis connection status")


class HealthResponse(BaseModel):
    """Health check response."""
    success: bool = True
    data: HealthData
    error: None = None
