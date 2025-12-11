---
sidebar_position: 2
---

# Memories API

Create, read, update, and delete memories.

## Create Memory

```http
POST /v1/memories
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Memory content (1-50,000 chars) |
| `metadata` | object | No | Key-value metadata |
| `memory_id` | string | No | Custom ID |

### Example

```bash
curl -X POST https://api.memory-api.com/v1/memories \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User works as a data scientist at TechCorp",
    "metadata": {"category": "professional"}
  }'
```

### Response

```json
{
  "success": true,
  "data": {
    "id": "mem_abc123",
    "text": "User works as a data scientist at TechCorp",
    "metadata": {"category": "professional"},
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
}
```

---

## List Memories

```http
GET /v1/memories
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `per_page` | int | 50 | Items per page (max 100) |

### Example

```bash
curl "https://api.memory-api.com/v1/memories?page=1&per_page=20" \
  -H "Authorization: Bearer $API_KEY"
```

---

## Get Memory

```http
GET /v1/memories/{memory_id}
```

### Example

```bash
curl https://api.memory-api.com/v1/memories/mem_abc123 \
  -H "Authorization: Bearer $API_KEY"
```

---

## Update Memory

```http
PUT /v1/memories/{memory_id}
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | No | Updated text |
| `metadata` | object | No | Updated metadata (replaces existing) |

### Example

```bash
curl -X PUT https://api.memory-api.com/v1/memories/mem_abc123 \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "User is a senior data scientist at TechCorp"}'
```

---

## Delete Memory

```http
DELETE /v1/memories/{memory_id}
```

### Example

```bash
curl -X DELETE https://api.memory-api.com/v1/memories/mem_abc123 \
  -H "Authorization: Bearer $API_KEY"
```

---

## Delete All Memories

```http
DELETE /v1/memories?confirm=true
```

:::danger Destructive Action
This permanently deletes all memories. The `confirm=true` parameter is required.
:::

### Example

```bash
curl -X DELETE "https://api.memory-api.com/v1/memories?confirm=true" \
  -H "Authorization: Bearer $API_KEY"
```
