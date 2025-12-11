---
sidebar_position: 1
---

# Quickstart

Get started with Memory API in 5 minutes.

## 1. Create an Account

Sign up to get your API key:

```bash
curl -X POST https://api.memory-api.com/v1/signup \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Company",
    "email": "you@example.com"
  }'
```

**Response:**
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

:::caution Save Your API Key
The full API key is only shown once. Save it securely!
:::

## 2. Store Your First Memory

```bash
export API_KEY="mem_live_your_api_key"

curl -X POST https://api.memory-api.com/v1/memories \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User prefers dark mode interfaces",
    "metadata": {"category": "preference"}
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "mem_xyz789",
    "text": "User prefers dark mode interfaces",
    "metadata": {"category": "preference"},
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

## 3. Query Your Memories

```bash
curl -X POST https://api.memory-api.com/v1/query \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "context": "What are the users UI preferences?"
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "memories": [
      {
        "id": "mem_xyz789",
        "text": "User prefers dark mode interfaces",
        "relevance_score": 0.92
      }
    ]
  }
}
```

## 4. Get an AI Answer

```bash
curl -X POST https://api.memory-api.com/v1/query/answer \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "context": "What theme should I use for the users dashboard?"
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "answer": "Based on the user's preferences, you should use a dark theme for the dashboard. The user has indicated they prefer dark mode interfaces.",
    "memories_used": [...]
  }
}
```

## Next Steps

- [Authentication](/getting-started/authentication) - Learn about API keys and scopes
- [Core Concepts](/getting-started/concepts) - Understand memories, queries, and retrieval
- [API Reference](/api-reference/overview) - Complete endpoint documentation
