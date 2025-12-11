---
sidebar_position: 3
---

# Query API

Find relevant memories and generate answers.

## Query Memories

Find memories relevant to a context.

```http
POST /v1/query
```

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `context` | string | Yes | - | Query context/question |
| `max_memories` | int | No | 20 | Max results (1-100) |
| `relevance_threshold` | float | No | 0.5 | Min score (0-1) |
| `include_metadata` | bool | No | true | Include metadata |

### Example

```bash
curl -X POST https://api.memory-api.com/v1/query \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "context": "What programming languages does the user know?",
    "max_memories": 5,
    "relevance_threshold": 0.6
  }'
```

### Response

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
      }
    ],
    "query_context": "What programming languages does the user know?",
    "total_memories_searched": 42
  }
}
```

---

## Query with Answer

Find memories and generate an AI answer.

```http
POST /v1/query/answer
```

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `context` | string | Yes | - | Question to answer |
| `max_memories` | int | No | 20 | Max memories to consider |
| `relevance_threshold` | float | No | 0.5 | Min relevance |
| `prompt_template` | string | No | null | Custom prompt |

### Example

```bash
curl -X POST https://api.memory-api.com/v1/query/answer \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "context": "What industry does the user work in?"
  }'
```

### Response

```json
{
  "success": true,
  "data": {
    "answer": "Based on the stored memories, the user works in the technology industry as a data scientist.",
    "memories_used": [...],
    "model": "gpt-4o-mini"
  }
}
```

---

## Extract Memories

Extract memories from a conversation or document.

```http
POST /v1/query/extract
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes | Content to extract from |
| `content_type` | string | No | Type: conversation, document, notes |
| `metadata` | object | No | Metadata for extracted memories |

### Example

```bash
curl -X POST https://api.memory-api.com/v1/query/extract \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User: Hi, I am John and I work at Google.\nAssistant: Nice to meet you!",
    "content_type": "conversation"
  }'
```

### Response

```json
{
  "success": true,
  "data": {
    "extracted_count": 2,
    "memory_ids": ["mem_new1", "mem_new2"]
  }
}
```
