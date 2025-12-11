"""
Application settings using Pydantic Settings.

All configuration is loaded from environment variables with sensible defaults.
For production, set these via environment variables or .env file.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Application
    # ==========================================================================
    app_name: str = "Memory API"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = Field(default="development", description="development|staging|production")

    # ==========================================================================
    # API Server
    # ==========================================================================
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    api_prefix: str = "/v1"

    # ==========================================================================
    # Database (PostgreSQL)
    # ==========================================================================
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/memory_api",
        description="PostgreSQL connection URL",
    )
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_echo: bool = False  # Log SQL queries

    # ==========================================================================
    # Redis
    # ==========================================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for caching and rate limiting",
    )
    redis_prefix: str = "memapi:"

    # ==========================================================================
    # Security
    # ==========================================================================
    secret_key: SecretStr = Field(
        default=SecretStr("CHANGE-ME-IN-PRODUCTION-use-openssl-rand-hex-32"),
        description="Secret key for signing tokens. Generate with: openssl rand -hex 32",
    )
    api_key_prefix: str = "mem"
    api_key_length: int = 32
    allowed_hosts: list[str] = ["*"]
    cors_origins: list[str] = ["*"]

    # ==========================================================================
    # Rate Limiting (requests per minute by tier)
    # ==========================================================================
    rate_limit_free: int = 10
    rate_limit_starter: int = 60
    rate_limit_professional: int = 300
    rate_limit_enterprise: int = 1000
    rate_limit_window_seconds: int = 60

    # ==========================================================================
    # OpenAI API (for memory system)
    # ==========================================================================
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key for LLM and embeddings",
    )
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_nano_model: str = "gpt-4o-mini"  # For relevance scoring

    # ==========================================================================
    # Memory System Defaults
    # ==========================================================================
    default_relevance_threshold: float = 0.5
    default_max_memories: int = 20
    default_prefilter_top_k: int = 100

    # ==========================================================================
    # Billing (Stripe)
    # ==========================================================================
    stripe_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Stripe secret API key",
    )
    stripe_webhook_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Stripe webhook signing secret",
    )
    stripe_price_starter: str = ""  # Stripe price ID for starter plan
    stripe_price_professional: str = ""  # Stripe price ID for professional plan

    # ==========================================================================
    # Usage Tracking
    # ==========================================================================
    usage_flush_interval_seconds: int = 60
    usage_batch_size: int = 100

    # ==========================================================================
    # Logging
    # ==========================================================================
    log_level: str = "INFO"
    log_format: str = "json"  # json or console

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
