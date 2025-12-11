---
sidebar_position: 1
---

# API Overview

The Memory API is a RESTful API that uses JSON for requests and responses.

## Base URL

```
https://api.memory-api.com/v1
```

## Authentication

Include your API key in every request:

```bash
Authorization: Bearer mem_live_your_api_key
```

## Endpoints Summary

### Memories
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/memories` | Create a memory |
| `GET` | `/memories` | List memories |
| `GET` | `/memories/{id}` | Get a memory |
| `PUT` | `/memories/{id}` | Update a memory |
| `DELETE` | `/memories/{id}` | Delete a memory |

### Query & Retrieval
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Find relevant memories |
| `POST` | `/query/answer` | Query with AI answer |
| `POST` | `/query/extract` | Extract memories from text |

### Account
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/account` | Get account details |
| `GET` | `/account/api-keys` | List API keys |
| `POST` | `/account/api-keys` | Create API key |
| `DELETE` | `/account/api-keys/{id}` | Revoke API key |

### Usage
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/usage` | Current usage |
| `GET` | `/usage/history` | Historical usage |
| `GET` | `/usage/projection` | Projected usage |

## Response Format

### Success Response
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

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message"
  }
}
```

## Rate Limit Headers

Every response includes:
- `X-RateLimit-Limit`: Your limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## Pagination

List endpoints support pagination:

```bash
GET /v1/memories?page=1&per_page=50
```

Response includes:
```json
{
  "pagination": {
    "total": 150,
    "page": 1,
    "per_page": 50,
    "total_pages": 3,
    "has_next": true
  }
}
```
