---
sidebar_position: 2
---

# Authentication

Learn how to authenticate with the Memory API.

## API Keys

All API requests (except `/health` and `/signup`) require authentication using an API key.

### Key Format

```
mem_{environment}_{random_string}
```

- `mem_live_...` - Production keys (counts against billing)
- `mem_test_...` - Sandbox keys (free, for testing)

### Using Your API Key

Include your key in every request using one of these methods:

**Authorization Header (Recommended)**
```bash
curl https://api.memory-api.com/v1/memories \
  -H "Authorization: Bearer mem_live_abc123..."
```

**X-API-Key Header**
```bash
curl https://api.memory-api.com/v1/memories \
  -H "X-API-Key: mem_live_abc123..."
```

## Managing API Keys

### List Keys

```bash
curl https://api.memory-api.com/v1/account/api-keys \
  -H "Authorization: Bearer $API_KEY"
```

### Create a New Key

```bash
curl -X POST https://api.memory-api.com/v1/account/api-keys \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Server",
    "environment": "live",
    "scopes": ["read", "write"]
  }'
```

### Revoke a Key

```bash
curl -X DELETE https://api.memory-api.com/v1/account/api-keys/key_abc123 \
  -H "Authorization: Bearer $API_KEY"
```

## Scopes

API keys can have different permission scopes:

| Scope | Permissions |
|-------|-------------|
| `read` | Read memories, query, view usage |
| `write` | Create and update memories |
| `delete` | Delete memories |
| `admin` | Manage API keys, full access |

### Example: Read-Only Key

```bash
curl -X POST https://api.memory-api.com/v1/account/api-keys \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"name": "Analytics", "scopes": ["read"]}'
```

## Key Expiration

You can create keys that automatically expire:

```bash
curl -X POST https://api.memory-api.com/v1/account/api-keys \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"name": "Temp Key", "expires_in_days": 30}'
```

## Security Best Practices

:::tip Best Practices
1. **Never commit API keys** to version control
2. **Use environment variables** to store keys
3. **Create separate keys** for different environments
4. **Use minimal scopes** - only grant permissions needed
5. **Rotate keys regularly** for production systems
6. **Monitor key usage** in your dashboard
:::

## Environment Variables

Store your API key in an environment variable:

```bash
# .env file (don't commit this!)
MEMORY_API_KEY=mem_live_your_key_here
```

```python
# Python
import os
api_key = os.environ.get('MEMORY_API_KEY')
```

```javascript
// JavaScript
const apiKey = process.env.MEMORY_API_KEY;
```
