# Getting Started with Memory API

Welcome to the Memory API! This guide will help you get started with storing and retrieving memories for your AI applications.

## Quick Start

### 1. Create an Account

Sign up to get your API key:

```bash
curl -X POST https://api.memory.example.com/v1/signup \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Company",
    "email": "you@example.com"
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "id": "key_abc123",
    "key": "mem_live_your_secret_key_here",
    "name": "Default Key"
  }
}
```

> **Important:** Save your API key! It's only shown once.

### 2. Store Your First Memory

```bash
curl -X POST https://api.memory.example.com/v1/memories \
  -H "Authorization: Bearer mem_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User prefers dark mode interfaces",
    "metadata": {"category": "preference"}
  }'
```

### 3. Query Your Memories

```bash
curl -X POST https://api.memory.example.com/v1/query \
  -H "Authorization: Bearer mem_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "context": "What are the users UI preferences?"
  }'
```

That's it! You've stored and retrieved your first memory.

---

## Authentication

Include your API key in every request using one of these methods:

**Authorization Header (Recommended):**
```
Authorization: Bearer mem_live_your_api_key
```

**X-API-Key Header:**
```
X-API-Key: mem_live_your_api_key
```

### API Key Environments

- `mem_live_...` - Production keys
- `mem_test_...` - Sandbox/testing keys

Test keys have the same functionality but don't count against billing.

---

## Core Concepts

### Memories

A memory is a piece of information you want your AI to remember:

```json
{
  "id": "mem_abc123",
  "text": "User works as a software engineer at TechCorp",
  "metadata": {
    "category": "professional",
    "source": "onboarding",
    "confidence": 0.95
  },
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Retrieval

When you query, the API finds memories relevant to your context:

```json
{
  "context": "What does the user do for work?",
  "memories": [
    {
      "text": "User works as a software engineer at TechCorp",
      "relevance_score": 0.92
    }
  ]
}
```

### Metadata

Use metadata to organize and filter memories:

```json
{
  "text": "User prefers Python",
  "metadata": {
    "category": "preference",
    "topic": "programming",
    "timestamp": "2024-01-15"
  }
}
```

---

## Common Operations

### Create Memory

```bash
POST /v1/memories
```

```json
{
  "text": "Memory content here",
  "metadata": {"key": "value"}
}
```

### List Memories

```bash
GET /v1/memories?page=1&per_page=50
```

### Get Single Memory

```bash
GET /v1/memories/{memory_id}
```

### Update Memory

```bash
PUT /v1/memories/{memory_id}
```

```json
{
  "text": "Updated content",
  "metadata": {"updated": true}
}
```

### Delete Memory

```bash
DELETE /v1/memories/{memory_id}
```

### Query Memories

```bash
POST /v1/query
```

```json
{
  "context": "Your question or context",
  "max_memories": 10,
  "relevance_threshold": 0.5
}
```

### Query with Answer

```bash
POST /v1/query/answer
```

```json
{
  "context": "What programming language does the user prefer?"
}
```

Response includes an AI-generated answer based on memories.

---

## Rate Limits

| Plan | Requests/min | Requests/day |
|------|--------------|--------------|
| Free | 10 | 100 |
| Starter | 60 | 5,000 |
| Professional | 300 | 50,000 |

Rate limit info is included in response headers:
- `X-RateLimit-Limit`: Your limit
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Unix timestamp when limit resets

---

## Error Handling

All errors follow this format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `MISSING_API_KEY` | 401 | No API key provided |
| `INVALID_API_KEY` | 401 | API key is invalid |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `MEMORY_NOT_FOUND` | 404 | Memory doesn't exist |
| `VALIDATION_ERROR` | 400 | Invalid request data |

---

## SDKs

### Python

```python
pip install memory-api
```

```python
from memory_api import MemoryClient

client = MemoryClient(api_key="mem_live_...")

# Create memory
memory = client.memories.create(
    text="User prefers dark mode",
    metadata={"category": "preference"}
)

# Query memories
results = client.query(
    context="What are the user's UI preferences?"
)

for memory in results.memories:
    print(f"[{memory.relevance:.2f}] {memory.text}")
```

### JavaScript

```bash
npm install @memory-api/sdk
```

```javascript
import { MemoryClient } from '@memory-api/sdk';

const client = new MemoryClient({ apiKey: 'mem_live_...' });

// Create memory
const memory = await client.memories.create({
  text: 'User prefers dark mode',
  metadata: { category: 'preference' }
});

// Query memories
const results = await client.query({
  context: 'What are the user\'s UI preferences?'
});
```

---

## Next Steps

- [API Reference](./API_REFERENCE.md) - Complete endpoint documentation
- [Best Practices](./BEST_PRACTICES.md) - Tips for production use
- [Pricing](./PRICING.md) - Plans and pricing details

## Support

- Documentation: https://docs.memory.example.com
- Email: support@memory.example.com
- Status: https://status.memory.example.com
