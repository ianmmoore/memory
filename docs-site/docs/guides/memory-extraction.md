---
sidebar_position: 1
---

# Memory Extraction

Automatically extract memories from conversations and documents.

## How It Works

The extraction endpoint uses AI to identify and store important information:

```
Conversation → AI Analysis → Extracted Memories
```

## Basic Extraction

```bash
curl -X POST https://api.memory-api.com/v1/query/extract \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "content": "User: Hi, I am Sarah and I work at Netflix as a product manager.",
    "content_type": "conversation"
  }'
```

**Extracted memories:**
- "User's name is Sarah"
- "User works at Netflix"
- "User is a product manager"

## Content Types

| Type | Best For |
|------|----------|
| `conversation` | Chat logs, transcripts |
| `document` | Articles, reports |
| `notes` | Meeting notes, summaries |

## Adding Metadata

Tag extracted memories with source information:

```json
{
  "content": "...",
  "metadata": {
    "source": "onboarding_call",
    "date": "2024-01-15",
    "session_id": "abc123"
  }
}
```

## Best Practices

1. **Clean input** - Remove irrelevant content before extraction
2. **Use metadata** - Track where memories came from
3. **Batch wisely** - Extract from complete conversations, not fragments
4. **Review results** - Periodically audit extracted memories
