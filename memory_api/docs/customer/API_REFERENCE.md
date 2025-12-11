# Memory API Reference

Complete reference for all Memory API endpoints.

**Base URL:** `https://api.memory.example.com/v1`

**Authentication:** All endpoints (except `/health` and `/signup`) require an API key.

---

## Memories

### Create Memory

Create a new memory.

```
POST /memories
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Memory content (1-50,000 chars) |
| `metadata` | object | No | Key-value metadata |
| `memory_id` | string | No | Custom ID (auto-generated if omitted) |

**Example:**

```bash
curl -X POST https://api.memory.example.com/v1/memories \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User works as a data scientist at TechCorp",
    "metadata": {
      "category": "professional",
      "confidence": 0.95
    }
  }'
```

**Response:** `201 Created`

```json
{
  "success": true,
  "data": {
    "id": "mem_abc123",
    "text": "User works as a data scientist at TechCorp",
    "metadata": {
      "category": "professional",
      "confidence": 0.95
    },
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  },
  "meta": {
    "request_id": "req_xyz789",
    "processing_time_ms": 45
  }
}
```

---

### List Memories

List all memories with pagination.

```
GET /memories
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `per_page` | int | 50 | Items per page (max 100) |

**Example:**

```bash
curl "https://api.memory.example.com/v1/memories?page=1&per_page=20" \
  -H "Authorization: Bearer $API_KEY"
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": [
    {
      "id": "mem_abc123",
      "text": "User works as a data scientist",
      "metadata": {"category": "professional"},
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "total": 42,
    "page": 1,
    "per_page": 20,
    "total_pages": 3,
    "has_next": true,
    "has_prev": false
  }
}
```

---

### Get Memory

Get a single memory by ID.

```
GET /memories/{memory_id}
```

**Example:**

```bash
curl "https://api.memory.example.com/v1/memories/mem_abc123" \
  -H "Authorization: Bearer $API_KEY"
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "id": "mem_abc123",
    "text": "User works as a data scientist",
    "metadata": {"category": "professional"},
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
}
```

**Errors:**
- `404 MEMORY_NOT_FOUND` - Memory doesn't exist

---

### Update Memory

Update an existing memory.

```
PUT /memories/{memory_id}
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | No | Updated text |
| `metadata` | object | No | Updated metadata (replaces existing) |

**Example:**

```bash
curl -X PUT "https://api.memory.example.com/v1/memories/mem_abc123" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User works as a senior data scientist at TechCorp",
    "metadata": {"category": "professional", "updated": true}
  }'
```

**Response:** `200 OK`

---

### Delete Memory

Delete a memory (soft delete).

```
DELETE /memories/{memory_id}
```

**Example:**

```bash
curl -X DELETE "https://api.memory.example.com/v1/memories/mem_abc123" \
  -H "Authorization: Bearer $API_KEY"
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "deleted": true,
    "memory_id": "mem_abc123"
  }
}
```

---

### Delete All Memories

Delete all memories (requires confirmation).

```
DELETE /memories?confirm=true
```

**Example:**

```bash
curl -X DELETE "https://api.memory.example.com/v1/memories?confirm=true" \
  -H "Authorization: Bearer $API_KEY"
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "deleted": true,
    "count": 42
  }
}
```

---

### Count Memories

Get the total memory count.

```
GET /memories/count
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "count": 42
  }
}
```

---

## Query & Retrieval

### Query Memories

Find memories relevant to a context.

```
POST /query
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `context` | string | Yes | - | Query context/question |
| `max_memories` | int | No | 20 | Max memories to return (1-100) |
| `relevance_threshold` | float | No | 0.5 | Min relevance score (0-1) |
| `include_metadata` | bool | No | true | Include metadata in results |
| `metadata_filter` | object | No | null | Filter by metadata fields |

**Example:**

```bash
curl -X POST "https://api.memory.example.com/v1/query" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "context": "What programming languages does the user know?",
    "max_memories": 5,
    "relevance_threshold": 0.6
  }'
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "memories": [
      {
        "id": "mem_abc123",
        "text": "User is proficient in Python and JavaScript",
        "relevance_score": 0.92,
        "relevance_reasoning": "Directly mentions programming languages",
        "metadata": {"category": "skills"}
      },
      {
        "id": "mem_def456",
        "text": "User prefers Python for data analysis",
        "relevance_score": 0.78,
        "relevance_reasoning": "Mentions Python specifically",
        "metadata": {"category": "preference"}
      }
    ],
    "query_context": "What programming languages does the user know?",
    "total_memories_searched": 42
  },
  "meta": {
    "request_id": "req_xyz789",
    "processing_time_ms": 234,
    "usage": {
      "memories_accessed": 2,
      "tokens_processed": 150
    }
  }
}
```

---

### Query with Answer

Find memories and generate an AI answer.

```
POST /query/answer
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `context` | string | Yes | - | Question to answer |
| `max_memories` | int | No | 20 | Max memories to consider |
| `relevance_threshold` | float | No | 0.5 | Min relevance score |
| `prompt_template` | string | No | null | Custom prompt template |
| `model` | string | No | gpt-4o-mini | Model for answer generation |

**Example:**

```bash
curl -X POST "https://api.memory.example.com/v1/query/answer" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "context": "What industry does the user work in?"
  }'
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "answer": "Based on the stored memories, the user works in the technology industry as a data scientist at TechCorp.",
    "memories_used": [
      {
        "id": "mem_abc123",
        "text": "User works as a data scientist at TechCorp",
        "relevance_score": 0.95
      }
    ],
    "confidence": 0.9,
    "model": "gpt-4o-mini"
  },
  "meta": {
    "usage": {
      "tokens_processed": 250,
      "model_calls": 1
    }
  }
}
```

---

### Extract Memories

Extract memories from a conversation or document.

```
POST /query/extract
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | - | Content to extract from |
| `content_type` | string | No | conversation | Type: conversation, document, notes |
| `metadata` | object | No | null | Metadata for extracted memories |

**Example:**

```bash
curl -X POST "https://api.memory.example.com/v1/query/extract" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User: Hi, I am John. I work at Google as a software engineer.\nAssistant: Nice to meet you!",
    "content_type": "conversation",
    "metadata": {"source": "chat_session_123"}
  }'
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "extracted_count": 2,
    "memory_ids": ["mem_new1", "mem_new2"]
  }
}
```

---

## Account

### Get Account

Get account details.

```
GET /account
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "org_id": "org_abc123",
    "name": "My Company",
    "email": "admin@mycompany.com",
    "plan": {
      "id": "starter",
      "name": "Starter",
      "base_price_cents": 4900
    },
    "is_active": true,
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

---

### List API Keys

List all API keys.

```
GET /account/api-keys
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": [
    {
      "id": "key_abc123",
      "name": "Production Key",
      "key_prefix": "mem_live_a1b2c3...",
      "environment": "live",
      "scopes": ["read", "write"],
      "is_active": true,
      "created_at": "2024-01-15T10:00:00Z",
      "last_used_at": "2024-01-15T15:30:00Z"
    }
  ]
}
```

---

### Create API Key

Create a new API key.

```
POST /account/api-keys
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | No | "Default Key" | Key name |
| `environment` | string | No | "live" | live or test |
| `scopes` | array | No | ["read", "write"] | Permissions |
| `expires_in_days` | int | No | null | Days until expiration |

**Example:**

```bash
curl -X POST "https://api.memory.example.com/v1/account/api-keys" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "CI/CD Key",
    "scopes": ["read"],
    "expires_in_days": 90
  }'
```

**Response:** `201 Created`

```json
{
  "success": true,
  "data": {
    "id": "key_new123",
    "name": "CI/CD Key",
    "key": "mem_live_full_key_shown_only_once",
    "key_prefix": "mem_live_full_key...",
    "scopes": ["read"],
    "expires_at": "2024-04-15T10:00:00Z"
  }
}
```

> **Warning:** The full `key` is only shown once. Save it securely!

---

### Revoke API Key

Revoke (disable) an API key.

```
DELETE /account/api-keys/{key_id}
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "revoked": true,
    "key_id": "key_abc123"
  }
}
```

---

## Usage

### Get Current Usage

Get usage for the current billing period.

```
GET /usage
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "current": {
      "period": "2024-01",
      "api_calls": 1523,
      "tokens_processed": 45230,
      "tokens_input": 30150,
      "tokens_output": 15080,
      "memories_stored": 156,
      "memories_read": 892,
      "memories_written": 45,
      "embeddings_generated": 45,
      "estimated_cost_cents": 125
    },
    "limits": {
      "api_calls": 10000,
      "tokens_processed": 1000000,
      "memories_stored": 10000
    },
    "percent_used": {
      "api_calls": 15.23,
      "tokens_processed": 4.52,
      "memories_stored": 1.56
    }
  }
}
```

---

### Get Usage History

Get historical usage data.

```
GET /usage/history?days=30
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "period_start": "2023-12-16",
    "period_end": "2024-01-15",
    "history": [
      {
        "date": "2024-01-15",
        "api_calls": 523,
        "tokens_total": 15230,
        "cost_cents": 42
      }
    ]
  }
}
```

---

### Get Usage Projection

Get projected month-end usage.

```
GET /usage/projection
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "period": "2024-01",
    "days_elapsed": 15,
    "days_remaining": 16,
    "current": {
      "api_calls": 1523,
      "tokens_total": 45230,
      "cost_cents": 125
    },
    "projected": {
      "api_calls": 3147,
      "tokens_total": 93475,
      "cost_cents": 258
    }
  }
}
```

---

### Export Usage

Export usage data.

```
GET /usage/export?format=csv&days=30
```

**Parameters:**
- `format`: csv or json
- `days`: Number of days (1-90)

**Response:** File download

---

## Health

### Health Check

Check service health (no authentication required).

```
GET /health
```

**Response:** `200 OK`

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "database": "healthy",
    "redis": "healthy"
  }
}
```

---

## Signup

### Create Account

Create a new account (no authentication required).

```
POST /signup
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Organization name |
| `email` | string | Yes | Contact email |
| `plan_id` | string | No | Initial plan (default: free) |

**Response:** `201 Created`

Returns an API key (see Create API Key response).

---

## Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `MISSING_API_KEY` | 401 | No API key provided |
| `INVALID_API_KEY` | 401 | Invalid or expired key |
| `INSUFFICIENT_PERMISSIONS` | 403 | Key lacks required scope |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `DAILY_QUOTA_EXCEEDED` | 429 | Daily limit reached |
| `MEMORY_NOT_FOUND` | 404 | Memory ID not found |
| `MEMORY_ID_EXISTS` | 409 | Custom ID already exists |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Response Headers

All responses include:

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier |
| `X-RateLimit-Limit` | Your rate limit |
| `X-RateLimit-Remaining` | Requests remaining |
| `X-RateLimit-Reset` | Unix timestamp when limit resets |
