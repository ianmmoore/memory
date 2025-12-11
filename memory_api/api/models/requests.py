"""
Pydantic request models for API validation.

All incoming request data is validated against these models.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re


class CreateMemoryRequest(BaseModel):
    """Request to create a new memory."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The memory text content",
        examples=["User prefers dark mode interfaces"],
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional metadata key-value pairs",
        examples=[{"category": "preference", "source": "onboarding"}],
    )
    memory_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Optional custom memory ID (auto-generated if not provided)",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Memory text cannot be empty or whitespace only")
        return v.strip()

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Optional[dict]) -> Optional[dict]:
        if v is None:
            return v
        # Ensure metadata values are JSON-serializable basic types
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError("Metadata keys must be strings")
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                raise ValueError(f"Metadata value for '{key}' is not JSON-serializable")
        return v


class UpdateMemoryRequest(BaseModel):
    """Request to update an existing memory."""

    model_config = ConfigDict(extra="forbid")

    text: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=50000,
        description="Updated memory text",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Updated metadata (replaces existing)",
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Memory text cannot be empty or whitespace only")
        return v.strip() if v else v


class QueryRequest(BaseModel):
    """Request to query for relevant memories."""

    model_config = ConfigDict(extra="forbid")

    context: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The context/question to find relevant memories for",
        examples=["What are the user's UI preferences?"],
    )
    max_memories: Optional[int] = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of memories to return",
    )
    relevance_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score (0-1) for returned memories",
    )
    include_metadata: Optional[bool] = Field(
        default=True,
        description="Include memory metadata in response",
    )
    metadata_filter: Optional[dict[str, Any]] = Field(
        default=None,
        description="Filter memories by metadata fields",
        examples=[{"category": "preference"}],
    )


class QueryWithAnswerRequest(BaseModel):
    """Request to query memories and generate an answer."""

    model_config = ConfigDict(extra="forbid")

    context: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The question or context to answer",
    )
    max_memories: Optional[int] = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum memories to consider for answering",
    )
    relevance_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score",
    )
    prompt_template: Optional[str] = Field(
        default=None,
        max_length=10000,
        description="Custom prompt template. Use {context} and {memories} placeholders.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use for answer generation (default: gpt-4o-mini)",
    )


class BatchCreateMemoriesRequest(BaseModel):
    """Request to create multiple memories at once."""

    model_config = ConfigDict(extra="forbid")

    memories: list[CreateMemoryRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of memories to create",
    )
    generate_embeddings: Optional[bool] = Field(
        default=True,
        description="Generate embeddings for memories immediately",
    )


class ExtractMemoriesRequest(BaseModel):
    """Request to extract memories from a conversation/document."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(
        ...,
        min_length=1,
        max_length=500000,
        description="The conversation or document to extract memories from",
    )
    content_type: Optional[str] = Field(
        default="conversation",
        description="Type of content: conversation, document, or notes",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Metadata to attach to all extracted memories",
    )


class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="Default Key",
        min_length=1,
        max_length=255,
        description="Human-readable name for the key",
    )
    environment: str = Field(
        default="live",
        pattern=r"^(live|test)$",
        description="Environment: 'live' for production, 'test' for sandbox",
    )
    scopes: Optional[list[str]] = Field(
        default=["read", "write"],
        description="Permission scopes for this key",
    )
    expires_in_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=365,
        description="Days until key expires (optional)",
    )

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, v: Optional[list[str]]) -> list[str]:
        valid_scopes = {"read", "write", "delete", "admin"}
        if v:
            for scope in v:
                if scope not in valid_scopes:
                    raise ValueError(f"Invalid scope: {scope}. Valid scopes: {valid_scopes}")
        return v or ["read", "write"]


class CreateOrganizationRequest(BaseModel):
    """Request to create a new organization (signup)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Organization name",
    )
    email: str = Field(
        ...,
        max_length=255,
        description="Contact email address",
    )
    plan_id: Optional[str] = Field(
        default="free",
        description="Initial pricing plan",
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        # Basic email validation
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError("Invalid email address format")
        return v.lower()
