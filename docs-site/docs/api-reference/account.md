---
sidebar_position: 4
---

# Account API

Manage your account and API keys.

## Get Account

```http
GET /v1/account
```

### Response

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

## List API Keys

```http
GET /v1/account/api-keys
```

### Response

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

## Create API Key

```http
POST /v1/account/api-keys
```

### Request Body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | "Default Key" | Key name |
| `environment` | string | "live" | live or test |
| `scopes` | array | ["read", "write"] | Permissions |
| `expires_in_days` | int | null | Days until expiration |

### Example

```bash
curl -X POST https://api.memory-api.com/v1/account/api-keys \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "CI/CD Key",
    "scopes": ["read"],
    "expires_in_days": 90
  }'
```

### Response

```json
{
  "success": true,
  "data": {
    "id": "key_new123",
    "name": "CI/CD Key",
    "key": "mem_live_full_key_shown_only_once",
    "scopes": ["read"],
    "expires_at": "2024-04-15T10:00:00Z"
  }
}
```

:::warning
The full `key` is only shown once. Save it securely!
:::

---

## Revoke API Key

```http
DELETE /v1/account/api-keys/{key_id}
```

### Example

```bash
curl -X DELETE https://api.memory-api.com/v1/account/api-keys/key_abc123 \
  -H "Authorization: Bearer $API_KEY"
```

---

## Signup (Create Account)

```http
POST /v1/signup
```

No authentication required.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Organization name |
| `email` | string | Yes | Contact email |
| `plan_id` | string | No | Initial plan (default: free) |

### Example

```bash
curl -X POST https://api.memory-api.com/v1/signup \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Company",
    "email": "admin@mycompany.com"
  }'
```
