# Memory API - Internal Architecture Documentation

## Overview

This document describes the internal architecture of the Memory API service for maintainers and developers.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Code Organization](#code-organization)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Security Model](#security-model)
6. [Billing System](#billing-system)
7. [Deployment](#deployment)
8. [Monitoring](#monitoring)
9. [Development Guide](#development-guide)

---

## System Architecture

### High-Level Architecture

```
                                    ┌─────────────────┐
                                    │   Cloudflare    │
                                    │   (DDoS/WAF)    │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │   Load Balancer │
                                    │   (ALB/nginx)   │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
           │   API Pod 1     │     │   API Pod 2     │     │   API Pod N     │
           │   (FastAPI)     │     │   (FastAPI)     │     │   (FastAPI)     │
           └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                    │                        │                        │
                    └────────────────────────┼────────────────────────┘
                                             │
         ┌───────────────┬───────────────────┼───────────────┬───────────────┐
         │               │                   │               │               │
┌────────▼────────┐ ┌────▼─────┐    ┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼──────┐
│   PostgreSQL    │ │  Redis   │    │    Celery     │ │ OpenAI    │ │   Stripe     │
│   (Primary DB)  │ │ (Cache)  │    │   (Workers)   │ │   API     │ │   (Billing)  │
└─────────────────┘ └──────────┘    └───────────────┘ └───────────┘ └──────────────┘
```

### Component Responsibilities

| Component | Purpose | Technology |
|-----------|---------|------------|
| **API Service** | HTTP request handling | FastAPI + Uvicorn |
| **PostgreSQL** | Persistent data storage | PostgreSQL 15 + pgvector |
| **Redis** | Caching, rate limiting, usage tracking | Redis 7 |
| **OpenAI API** | LLM inference, embeddings | OpenAI Python SDK |
| **Stripe** | Payment processing | Stripe Python SDK |

---

## Code Organization

```
memory_api/
├── api/                          # FastAPI application
│   ├── __init__.py
│   ├── main.py                   # Application factory & setup
│   ├── routes/                   # API endpoints
│   │   ├── memories.py           # Memory CRUD
│   │   ├── query.py              # Retrieval & answering
│   │   ├── account.py            # Account management
│   │   ├── usage.py              # Usage & billing
│   │   └── health.py             # Health checks
│   ├── middleware/               # Request processing
│   │   ├── auth.py               # API key authentication
│   │   ├── rate_limit.py         # Rate limiting
│   │   └── usage_tracking.py     # Usage metering
│   └── models/                   # Pydantic schemas
│       ├── requests.py           # Request validation
│       └── responses.py          # Response formatting
│
├── billing/                      # Billing system
│   ├── pricing.py                # Pricing engine
│   └── usage.py                  # Usage aggregation
│
├── config/                       # Configuration
│   ├── settings.py               # Pydantic settings
│   └── pricing_config.yaml       # Pricing definitions
│
├── db/                           # Database
│   ├── database.py               # Connection management
│   ├── models.py                 # SQLAlchemy models
│   └── init.sql                  # Initial schema
│
├── tests/                        # Test suite
│   ├── conftest.py               # Fixtures
│   ├── test_auth.py
│   ├── test_memories.py
│   ├── test_query.py
│   └── ...
│
├── docs/                         # Documentation
│   ├── internal/                 # This documentation
│   └── customer/                 # Customer docs
│
├── Dockerfile                    # Container build
├── docker-compose.yml            # Local development
└── requirements.txt              # Dependencies
```

### Separation of Concerns

The codebase is designed with clear separation:

1. **API Layer** (`api/`) - HTTP handling, validation, serialization
2. **Business Logic** (`billing/`) - Pricing, usage calculations
3. **Data Layer** (`db/`) - Persistence, models
4. **Configuration** (`config/`) - Settings, pricing
5. **Core Science** (`../memory_lib/`) - Memory system algorithms (separate package)

The memory system's "science" (relevance scoring, embedding prefiltering, etc.) lives in `memory_lib/` and is imported as a dependency. This keeps the API boilerplate separate from the core algorithms.

---

## Core Components

### 1. Authentication (`api/middleware/auth.py`)

**API Key Format:**
```
mem_{environment}_{random_hex}
Example: mem_live_a1b2c3d4e5f6g7h8...
```

**Key Storage:**
- Full key: Never stored (only shown once on creation)
- Key hash: bcrypt hash for verification
- Key prefix: First 12 chars for lookup

**Authentication Flow:**
```python
# 1. Extract key from header
api_key = request.headers.get("Authorization")  # Bearer token
# or
api_key = request.headers.get("X-API-Key")

# 2. Look up by prefix (fast index lookup)
key_record = db.query(APIKey).filter_by(key_prefix=prefix).first()

# 3. Verify with bcrypt (secure comparison)
if bcrypt.verify(api_key, key_record.key_hash):
    # Authenticated!
```

### 2. Rate Limiting (`api/middleware/rate_limit.py`)

**Algorithm:** Sliding window using Redis sorted sets

```lua
-- Lua script for atomic rate limiting
local key = KEYS[1]
local now = ARGV[1]
local window = ARGV[2]
local limit = ARGV[3]

-- Remove old entries
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)

-- Count current requests
local current = redis.call('ZCARD', key)

if current < limit then
    -- Add request timestamp
    redis.call('ZADD', key, now, now .. '-' .. random())
    redis.call('EXPIRE', key, window)
    return {1, current + 1, limit}  -- Allowed
else
    return {0, current, limit}  -- Denied
end
```

**Rate Limits by Plan:**
| Plan | Per Minute | Per Day | Concurrent |
|------|------------|---------|------------|
| Free | 10 | 100 | 2 |
| Starter | 60 | 5,000 | 5 |
| Professional | 300 | 50,000 | 20 |
| Enterprise | 1000 | Unlimited | 100 |

### 3. Usage Tracking (`api/middleware/usage_tracking.py`)

**Two-tier tracking system:**

1. **Real-time (Redis):**
   - Immediate counters for quota checks
   - Per-organization hash: `memapi:usage:realtime:{org_id}:{period}`
   - Updated on every request

2. **Historical (PostgreSQL):**
   - Buffered writes via Redis list
   - Background worker flushes to `usage_events` table
   - Daily rollups for billing queries

**Tracked Metrics:**
- `api_calls`: Total requests
- `tokens_input`: LLM input tokens
- `tokens_output`: LLM output tokens
- `memories_read`: Memories retrieved
- `memories_written`: Memories created/updated
- `embeddings_generated`: Embedding vectors created

### 4. Memory System Integration (`api/routes/query.py`)

The API connects to the core memory system via an adapter:

```python
class APIMemoryAdapter:
    """Bridges API database with memory_lib algorithms."""

    async def retrieve_relevant_memories(
        self,
        context: str,
        max_memories: int,
        relevance_threshold: float,
    ) -> list[dict]:
        # 1. Get all memories from PostgreSQL
        memories = await self.get_all_memories()

        # 2. Apply memory_lib's retrieval algorithm
        # (embedding prefilter + LLM-based scoring)
        scored = await memory_lib.score_relevance(
            context, memories
        )

        # 3. Filter and return
        return [m for m in scored if m.score >= threshold][:max_memories]
```

---

## Data Flow

### Request Lifecycle

```
1. Request arrives at API
       │
       ▼
2. UsageTrackingMiddleware
   - Start timer
   - Generate request ID
       │
       ▼
3. AuthMiddleware
   - Extract API key
   - Validate and load org
       │
       ▼
4. Route Handler
   - Validate request (Pydantic)
   - Check rate limits
   - Execute business logic
   - Track detailed usage
       │
       ▼
5. Response
   - Format response
   - Add rate limit headers
   - Log completion
```

### Memory Query Flow

```
User Query: "What does the user do for work?"
       │
       ▼
┌──────────────────────────────────────────┐
│  1. Retrieve all memories for org        │
│     (PostgreSQL: SELECT * FROM memories  │
│      WHERE org_id = ?)                   │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  2. Embedding Prefilter (Optional)       │
│     - Generate query embedding           │
│     - Cosine similarity ranking          │
│     - Select top-K candidates            │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  3. LLM Relevance Scoring                │
│     - Score each candidate with LLM      │
│     - Get relevance 0-1 with reasoning   │
│     - Filter by threshold                │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  4. Answer Generation (if requested)     │
│     - Build prompt with memories         │
│     - Call LLM for answer                │
│     - Return with citations              │
└──────────────────────────────────────────┘
```

---

## Security Model

### Multi-tenancy

All data is isolated by `org_id`:
- Every query includes `WHERE org_id = ?`
- API keys belong to organizations
- No cross-tenant data access possible

### Secrets Management

```python
# Settings use Pydantic SecretStr
class Settings:
    secret_key: SecretStr
    openai_api_key: SecretStr
    stripe_api_key: SecretStr

# Access via .get_secret_value()
# Never logged or serialized
```

### Input Validation

All inputs validated via Pydantic:
```python
class CreateMemoryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)
    metadata: Optional[dict] = None

    @field_validator("text")
    def validate_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Cannot be empty")
        return v.strip()
```

### Audit Logging

Security-sensitive operations logged to `audit_logs`:
- API key creation/revocation
- Organization changes
- Admin actions

---

## Billing System

### Pricing Architecture

```yaml
# config/pricing_config.yaml
plans:
  starter:
    base_price_cents: 4900
    limits:
      api_calls: 10000
      tokens_processed: 1000000
    overage_rates:
      api_calls:
        price_cents: 50
        per_units: 1000
```

### Invoice Calculation

```python
async def calculate_invoice(org_id, usage, period):
    pricing = await get_effective_pricing(org_id)

    line_items = []

    # 1. Base subscription
    line_items.append(pricing.plan.base_price)

    # 2. Overage charges
    for metric, amount in usage.items():
        if amount > pricing.limits[metric]:
            overage = amount - pricing.limits[metric]
            charge = overage * pricing.rates[metric]
            # Apply volume discount
            charge *= (1 - get_volume_discount(metric, amount))
            line_items.append(charge)

    # 3. Add-ons
    for addon in pricing.addons:
        line_items.append(addon.price)

    # 4. Apply credits
    subtotal = sum(line_items)
    credits = min(pricing.credits, subtotal)

    return Invoice(
        subtotal=subtotal,
        credits_applied=credits,
        total=subtotal - credits,
    )
```

---

## Deployment

### Docker Build

```bash
# Build production image
docker build -t memory-api:latest --target production .

# Build development image (with hot reload)
docker build -t memory-api:dev --target development .
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `REDIS_URL` | Redis connection string | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `SECRET_KEY` | App secret for signing | Yes |
| `STRIPE_API_KEY` | Stripe API key | No |
| `ENVIRONMENT` | development/staging/production | No |

### Health Checks

```bash
# Liveness probe
curl http://localhost:8000/health

# Response
{
  "data": {
    "status": "healthy",
    "database": "healthy",
    "redis": "healthy"
  }
}
```

---

## Monitoring

### Metrics to Track

1. **Request Metrics:**
   - Request rate (by endpoint)
   - Latency percentiles (p50, p95, p99)
   - Error rate

2. **Business Metrics:**
   - Active organizations
   - API calls per org
   - Revenue by plan

3. **System Metrics:**
   - CPU/Memory usage
   - Database connections
   - Redis memory

### Logging

Structured JSON logging via `structlog`:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "info",
  "event": "request_completed",
  "request_id": "abc123",
  "org_id": "org_xyz",
  "endpoint": "/v1/memories",
  "status_code": 200,
  "duration_ms": 45
}
```

---

## Development Guide

### Local Setup

```bash
# Start dependencies
docker-compose up -d db redis

# Install Python dependencies
pip install -r requirements.txt

# Run migrations
python -c "from db.database import init_db; import asyncio; asyncio.run(init_db())"

# Start development server
python -m uvicorn api.main:app --reload
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=api --cov=billing --cov-report=html

# Specific test file
pytest tests/test_memories.py -v

# Watch mode
pytest-watch
```

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy api/
```

### Adding New Endpoints

1. Create Pydantic models in `api/models/`
2. Add route handler in `api/routes/`
3. Register router in `api/main.py`
4. Add tests in `tests/`
5. Update OpenAPI docs

---

## Troubleshooting

### Common Issues

**Rate limit always fails:**
- Check Redis connection
- Verify `REDIS_URL` environment variable
- Ensure Redis is running: `redis-cli ping`

**Database connection errors:**
- Check `DATABASE_URL` format
- Ensure PostgreSQL is running
- Verify network connectivity

**Authentication fails:**
- Check API key format (must start with `mem_`)
- Verify key is active in database
- Check key hasn't expired

### Debug Mode

Enable detailed logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export LOG_FORMAT=console
```
