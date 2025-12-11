---
sidebar_position: 3
---

# Best Practices

Production-ready tips for using Memory API.

## Memory Design

### Keep Memories Atomic

```
❌ "User is John, works at Google, likes Python, lives in NYC"
✅ "User's name is John"
✅ "User works at Google"
✅ "User prefers Python"
✅ "User lives in NYC"
```

### Use Consistent Metadata

```json
{
  "category": "preference",
  "source": "onboarding",
  "confidence": 0.95,
  "timestamp": "2024-01-15"
}
```

## Security

### Protect API Keys

```bash
# ❌ Don't commit keys
API_KEY=mem_live_abc123

# ✅ Use environment variables
export MEMORY_API_KEY=$API_KEY
```

### Use Minimal Scopes

```json
{
  "name": "Read-Only Analytics",
  "scopes": ["read"]
}
```

### Rotate Keys Regularly

Create new keys periodically and revoke old ones.

## Performance

### Batch Operations

```python
# ❌ Many individual requests
for text in texts:
    client.memories.create(text=text)

# ✅ Use extraction for bulk
client.extract(content="\n".join(texts))
```

### Cache Results

```python
@cache(ttl=300)
def get_user_preferences(user_id):
    return client.query(context=f"preferences for {user_id}")
```

### Handle Rate Limits

```python
import time

try:
    result = client.query(context="...")
except RateLimitError as e:
    time.sleep(e.retry_after)
    result = client.query(context="...")
```

## Monitoring

### Track Usage

Check `/v1/usage` regularly:
- Monitor quota consumption
- Set up alerts at 80% usage
- Review projected costs

### Log Request IDs

Every response includes `X-Request-ID`. Log it for debugging:

```python
response = client.memories.create(text="...")
logger.info(f"Created memory, request_id: {response.meta.request_id}")
```

## Data Quality

### Review Extracted Memories

Periodically audit auto-extracted memories for accuracy.

### Update Stale Data

Mark or remove outdated memories:

```python
# Update with new info
client.memories.update(memory_id, text="User now works at Meta")
```

### Use Timestamps

Include timestamps in metadata to track freshness:

```json
{
  "text": "User prefers dark mode",
  "metadata": {
    "recorded_at": "2024-01-15T10:30:00Z"
  }
}
```
