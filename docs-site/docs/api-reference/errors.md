---
sidebar_position: 6
---

# Error Codes

All API errors follow a consistent format.

## Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description",
    "details": { ... }
  }
}
```

## Authentication Errors

| Code | HTTP | Description |
|------|------|-------------|
| `MISSING_API_KEY` | 401 | No API key provided |
| `INVALID_API_KEY` | 401 | API key is invalid |
| `INVALID_API_KEY_FORMAT` | 401 | Key doesn't match expected format |
| `API_KEY_EXPIRED` | 401 | API key has expired |
| `API_KEY_DISABLED` | 401 | API key has been revoked |

## Authorization Errors

| Code | HTTP | Description |
|------|------|-------------|
| `INSUFFICIENT_PERMISSIONS` | 403 | Key lacks required scope |
| `ORGANIZATION_DISABLED` | 403 | Organization is inactive |

## Rate Limiting Errors

| Code | HTTP | Description |
|------|------|-------------|
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests per minute |
| `DAILY_QUOTA_EXCEEDED` | 429 | Daily limit reached |
| `CONCURRENT_LIMIT_EXCEEDED` | 429 | Too many concurrent requests |

When rate limited, check these headers:
- `Retry-After`: Seconds to wait
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Resource Errors

| Code | HTTP | Description |
|------|------|-------------|
| `MEMORY_NOT_FOUND` | 404 | Memory ID doesn't exist |
| `MEMORY_ID_EXISTS` | 409 | Custom ID already taken |
| `API_KEY_NOT_FOUND` | 404 | API key ID not found |
| `ORGANIZATION_NOT_FOUND` | 404 | Organization not found |

## Validation Errors

| Code | HTTP | Description |
|------|------|-------------|
| `VALIDATION_ERROR` | 400 | Request body validation failed |
| `CONFIRMATION_REQUIRED` | 400 | Destructive action needs confirm=true |

Validation errors include details:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": {
      "field": "text",
      "errors": [
        {"field": "text", "message": "field required"}
      ]
    }
  }
}
```

## Server Errors

| Code | HTTP | Description |
|------|------|-------------|
| `INTERNAL_ERROR` | 500 | Unexpected server error |
| `EXTRACTION_FAILED` | 500 | Memory extraction failed |

## Handling Errors

### Python

```python
import requests

response = requests.post(url, json=data, headers=headers)

if not response.ok:
    error = response.json()["error"]
    if error["code"] == "RATE_LIMIT_EXCEEDED":
        time.sleep(int(response.headers["Retry-After"]))
        # Retry request
    elif error["code"] == "MEMORY_NOT_FOUND":
        # Handle missing resource
        pass
```

### JavaScript

```javascript
try {
  const response = await fetch(url, options);
  const data = await response.json();

  if (!data.success) {
    if (data.error.code === 'RATE_LIMIT_EXCEEDED') {
      const retryAfter = response.headers.get('Retry-After');
      await sleep(retryAfter * 1000);
      // Retry
    }
  }
} catch (err) {
  console.error('Request failed:', err);
}
```
