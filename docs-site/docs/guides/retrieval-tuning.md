---
sidebar_position: 2
---

# Retrieval Tuning

Optimize memory retrieval for your use case.

## Relevance Threshold

Control result quality vs. quantity:

```json
{
  "context": "user preferences",
  "relevance_threshold": 0.7
}
```

| Threshold | Results |
|-----------|---------|
| 0.3-0.5 | More results, lower precision |
| 0.5-0.7 | Balanced |
| 0.7-0.9 | Fewer results, higher precision |

## Max Memories

Limit the number of results:

```json
{
  "context": "...",
  "max_memories": 5
}
```

Use fewer memories for:
- Quick lookups
- Specific questions

Use more memories for:
- Comprehensive answers
- Context building

## Query Phrasing

How you phrase queries affects results:

| Query Style | Best For |
|------------|----------|
| Questions | "What does the user prefer?" |
| Keywords | "user preferences dark mode" |
| Statements | "Finding information about user preferences" |

## Metadata Filtering

Pre-filter memories by metadata:

```json
{
  "context": "preferences",
  "metadata_filter": {
    "category": "preference"
  }
}
```

## Tips

1. **Be specific** - Detailed queries get better results
2. **Use context** - Include relevant background information
3. **Test thresholds** - Find the right balance for your data
4. **Monitor relevance** - Check relevance_score in responses
