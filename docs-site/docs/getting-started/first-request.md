---
sidebar_position: 3
---

# Your First Request

Learn the basics of making API requests.

## Request Format

All requests use JSON and require these headers:

```bash
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

## Response Format

All responses follow this structure:

```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "request_id": "req_abc123",
    "processing_time_ms": 45
  }
}
```

## Create a Memory

```bash
curl -X POST https://api.memory-api.com/v1/memories \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User works as a software engineer at TechCorp",
    "metadata": {
      "category": "professional",
      "source": "onboarding"
    }
  }'
```

## List Memories

```bash
curl https://api.memory-api.com/v1/memories \
  -H "Authorization: Bearer $API_KEY"
```

## Query Memories

```bash
curl -X POST https://api.memory-api.com/v1/query \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "context": "What does the user do for work?",
    "max_memories": 5
  }'
```

## Error Handling

Errors return a consistent format:

```json
{
  "success": false,
  "error": {
    "code": "MEMORY_NOT_FOUND",
    "message": "Memory 'mem_xyz' not found"
  }
}
```

See [Error Codes](/api-reference/errors) for all error types.
