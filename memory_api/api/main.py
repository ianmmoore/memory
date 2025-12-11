"""
Memory API - FastAPI Application

This is the main entry point for the Memory API service.
It sets up the FastAPI application with all middleware, routes, and lifecycle events.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import structlog

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from db.database import init_db, close_db
from api.routes import (
    memories_router,
    query_router,
    account_router,
    usage_router,
    health_router,
)
from api.routes.account import signup_router
from api.middleware.auth import AuthMiddleware
from api.middleware.rate_limit import RateLimitMiddleware, rate_limiter
from api.middleware.usage_tracking import (
    UsageTrackingMiddleware,
    usage_tracker,
    UsageFlushWorker,
)


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Initialize database
    - Connect to Redis
    - Start background workers
    """
    # Startup
    logger.info("starting_application", version=settings.app_version)

    # Initialize database
    try:
        await init_db()
        logger.info("database_initialized")
    except Exception as e:
        logger.error("database_init_failed", error=str(e))
        # Continue anyway for development

    # Start usage flush worker (in production, run as separate process)
    flush_worker = UsageFlushWorker()
    import asyncio
    flush_task = asyncio.create_task(flush_worker.start())

    yield

    # Shutdown
    logger.info("shutting_down_application")

    # Stop flush worker
    flush_worker.stop()
    flush_task.cancel()

    # Close connections
    await rate_limiter.close()
    await usage_tracker.close()
    await close_db()

    logger.info("application_shutdown_complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory function allows creating multiple app instances
    for testing or different configurations.
    """
    app = FastAPI(
        title=settings.app_name,
        description="""
# Memory API

Intelligent memory storage and retrieval for AI applications.

## Features

- **Memory Storage**: Store and organize memories with metadata
- **Semantic Retrieval**: Find relevant memories using AI-powered search
- **Answer Generation**: Generate answers based on stored memories
- **Usage Tracking**: Monitor API usage and costs
- **Multi-tenant**: Secure isolation between organizations

## Authentication

All endpoints (except /health and /signup) require an API key.

Include your API key in the `Authorization` header:
```
Authorization: Bearer mem_live_your_api_key_here
```

Or in the `X-API-Key` header:
```
X-API-Key: mem_live_your_api_key_here
```

## Rate Limits

Rate limits vary by plan:
- Free: 10 requests/minute
- Starter: 60 requests/minute
- Professional: 300 requests/minute
- Enterprise: Custom

## Error Handling

All errors return a consistent format:
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message"
  }
}
```
        """,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ==========================================================================
    # Middleware (order matters - last added is first executed)
    # ==========================================================================

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
    )

    # Usage tracking (tracks request timing and basic usage)
    app.add_middleware(UsageTrackingMiddleware)

    # Rate limiting headers
    app.add_middleware(RateLimitMiddleware)

    # Authentication (adds org_id to request state)
    app.add_middleware(AuthMiddleware)

    # ==========================================================================
    # Exception Handlers
    # ==========================================================================

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors."""
        errors = exc.errors()
        first_error = errors[0] if errors else {}

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": first_error.get("msg", "Validation error"),
                    "details": {
                        "field": ".".join(str(loc) for loc in first_error.get("loc", [])),
                        "errors": [
                            {
                                "field": ".".join(str(loc) for loc in e.get("loc", [])),
                                "message": e.get("msg"),
                                "type": e.get("type"),
                            }
                            for e in errors
                        ],
                    },
                },
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(
            "unhandled_exception",
            error=str(exc),
            path=request.url.path,
            method=request.method,
        )

        # Don't expose internal errors in production
        if settings.is_production:
            message = "An internal error occurred"
        else:
            message = str(exc)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": message,
                },
            },
        )

    # ==========================================================================
    # Routes
    # ==========================================================================

    # Health check (no prefix, no auth)
    app.include_router(health_router)

    # Signup (no auth required)
    app.include_router(signup_router, prefix=settings.api_prefix)

    # Protected routes
    app.include_router(memories_router, prefix=settings.api_prefix)
    app.include_router(query_router, prefix=settings.api_prefix)
    app.include_router(account_router, prefix=settings.api_prefix)
    app.include_router(usage_router, prefix=settings.api_prefix)

    return app


# Create the default app instance
app = create_app()


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        workers=1 if settings.is_development else settings.workers,
        log_level=settings.log_level.lower(),
    )
