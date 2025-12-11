---
slug: /
sidebar_position: 1
---

# Memory API Documentation

Welcome to the Memory API documentation. Memory API provides intelligent memory storage and retrieval for AI applications.

## What is Memory API?

Memory API enables your AI applications to:

- **Store memories** - Save important information with metadata
- **Retrieve intelligently** - Find relevant memories using semantic search
- **Generate answers** - Get AI-powered answers based on stored memories
- **Extract knowledge** - Automatically extract memories from conversations

## Quick Example

```bash
# Store a memory
curl -X POST https://api.memory-api.com/v1/memories \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers dark mode interfaces"}'

# Query memories
curl -X POST https://api.memory-api.com/v1/query \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"context": "What are the users UI preferences?"}'
```

## Getting Started

<div className="row">
  <div className="col col--6">
    <div className="card margin-bottom--lg">
      <div className="card__header">
        <h3>🚀 Quickstart</h3>
      </div>
      <div className="card__body">
        <p>Get up and running in 5 minutes.</p>
      </div>
      <div className="card__footer">
        <a className="button button--primary button--block" href="/getting-started/quickstart">Start Building</a>
      </div>
    </div>
  </div>
  <div className="col col--6">
    <div className="card margin-bottom--lg">
      <div className="card__header">
        <h3>📖 API Reference</h3>
      </div>
      <div className="card__body">
        <p>Complete endpoint documentation.</p>
      </div>
      <div className="card__footer">
        <a className="button button--secondary button--block" href="/api-reference/overview">View Reference</a>
      </div>
    </div>
  </div>
</div>

## Key Features

| Feature | Description |
|---------|-------------|
| **Semantic Search** | AI-powered relevance scoring finds the right memories |
| **Multi-tenant** | Secure isolation between organizations |
| **Real-time Usage** | Monitor API usage and costs in real-time |
| **Extensible** | Custom metadata, webhooks, and integrations |

## SDKs

Install our official SDKs:

```bash
# Python
pip install memory-api

# JavaScript
npm install @memory-api/sdk
```

## Need Help?

- **Documentation** - You're in the right place!
- **API Status** - [status.memory-api.com](https://status.memory-api.com)
- **Support** - [support@memory-api.com](mailto:support@memory-api.com)
