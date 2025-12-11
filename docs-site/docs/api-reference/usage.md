---
sidebar_position: 5
---

# Usage API

Monitor your API usage and costs.

## Get Current Usage

```http
GET /v1/usage
```

### Response

```json
{
  "success": true,
  "data": {
    "current": {
      "period": "2024-01",
      "api_calls": 1523,
      "tokens_processed": 45230,
      "memories_stored": 156,
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

## Get Usage History

```http
GET /v1/usage/history?days=30
```

### Query Parameters

| Parameter | Default | Max | Description |
|-----------|---------|-----|-------------|
| `days` | 30 | 90 | Days of history |

### Response

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

## Get Usage Projection

```http
GET /v1/usage/projection
```

### Response

```json
{
  "success": true,
  "data": {
    "period": "2024-01",
    "days_elapsed": 15,
    "days_remaining": 16,
    "current": {
      "api_calls": 1523,
      "cost_cents": 125
    },
    "projected": {
      "api_calls": 3147,
      "cost_cents": 258
    }
  }
}
```

---

## Export Usage

```http
GET /v1/usage/export?format=csv&days=30
```

### Query Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `format` | csv, json | Export format |
| `days` | 1-90 | Days to export |

Returns a downloadable file.
