---
sidebar_position: 4
---

# Core Concepts

Understand the key concepts in Memory API.

## Memories

A **memory** is a piece of information you want your AI to remember.

```json
{
  "id": "mem_abc123",
  "text": "User prefers Python for data analysis",
  "metadata": {
    "category": "preference",
    "confidence": 0.95
  },
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Memory Components

| Field | Description |
|-------|-------------|
| `text` | The memory content (required) |
| `metadata` | Key-value pairs for organization |
| `id` | Unique identifier (auto-generated or custom) |

## Retrieval

**Retrieval** finds memories relevant to a given context using AI-powered semantic search.

```
Query: "What programming language does the user prefer?"
    ↓
Relevance Scoring (AI)
    ↓
Returns: "User prefers Python for data analysis" (score: 0.92)
```

### Relevance Scoring

Each memory gets a relevance score from 0 to 1:
- **0.9-1.0**: Highly relevant
- **0.7-0.9**: Relevant
- **0.5-0.7**: Possibly relevant
- **< 0.5**: Likely not relevant

## Answer Generation

The API can generate **answers** based on retrieved memories:

```
Question: "What should I use for the user's data project?"
    ↓
Retrieve relevant memories
    ↓
Generate answer: "Based on the user's preferences,
                  Python would be ideal for their data project."
```

## Organizations

An **organization** is your account. All data is isolated per organization.

- Each org has its own memories
- Multiple API keys per org
- Separate billing and usage

## API Keys

**API keys** authenticate requests and define permissions.

- `mem_live_...` - Production (billed)
- `mem_test_...` - Testing (free)

Keys have **scopes**: `read`, `write`, `delete`, `admin`

## Rate Limits

Requests are limited per plan:

| Plan | Requests/min |
|------|--------------|
| Free | 10 |
| Starter | 60 |
| Professional | 300 |

## Metadata

Use **metadata** to organize and filter memories:

```json
{
  "text": "User completed onboarding",
  "metadata": {
    "category": "event",
    "timestamp": "2024-01-15",
    "importance": "high"
  }
}
```

Best practices:
- Use consistent key names
- Keep values simple (strings, numbers, booleans)
- Don't store sensitive data in metadata
