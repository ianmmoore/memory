---
sidebar_position: 2
---

# JavaScript SDK

Official JavaScript/TypeScript SDK for Memory API.

## Installation

```bash
npm install @memory-api/sdk
# or
yarn add @memory-api/sdk
```

## Quick Start

```javascript
import { MemoryClient } from '@memory-api/sdk';

const client = new MemoryClient({ apiKey: 'mem_live_your_key' });

// Create a memory
const memory = await client.memories.create({
  text: 'User prefers dark mode interfaces',
  metadata: { category: 'preference' }
});

// Query memories
const results = await client.query({
  context: 'What are the user\'s UI preferences?'
});

results.memories.forEach(mem => {
  console.log(`[${mem.relevanceScore.toFixed(2)}] ${mem.text}`);
});
```

## TypeScript Support

Full TypeScript support with type definitions:

```typescript
import { MemoryClient, Memory, QueryResult } from '@memory-api/sdk';

const client = new MemoryClient({ apiKey: 'mem_live_...' });

const memory: Memory = await client.memories.create({
  text: 'TypeScript memory',
});

const results: QueryResult = await client.query({
  context: 'test query'
});
```

## Memory Operations

```javascript
// Create
const memory = await client.memories.create({
  text: 'Content',
  metadata: { key: 'value' }
});

// List
const list = await client.memories.list({ page: 1, perPage: 50 });

// Get
const mem = await client.memories.get('mem_abc123');

// Update
await client.memories.update('mem_abc123', {
  text: 'Updated'
});

// Delete
await client.memories.delete('mem_abc123');
```

## Query Operations

```javascript
// Basic query
const results = await client.query({
  context: 'What does the user like?',
  maxMemories: 10,
  relevanceThreshold: 0.5
});

// Query with answer
const answer = await client.queryAnswer({
  context: 'What theme should I use?'
});
console.log(answer.answer);

// Extract memories
const extracted = await client.extract({
  content: 'User: Hi, I\'m John from NYC',
  contentType: 'conversation'
});
```

## Error Handling

```javascript
import {
  MemoryClient,
  AuthError,
  RateLimitError,
  NotFoundError
} from '@memory-api/sdk';

try {
  const memory = await client.memories.get('mem_xyz');
} catch (err) {
  if (err instanceof NotFoundError) {
    console.log('Memory not found');
  } else if (err instanceof RateLimitError) {
    console.log(`Retry after ${err.retryAfter}s`);
  } else if (err instanceof AuthError) {
    console.log('Invalid API key');
  }
}
```

## Browser Usage

```html
<script type="module">
  import { MemoryClient } from 'https://cdn.memory-api.com/sdk.js';

  const client = new MemoryClient({ apiKey: 'mem_live_...' });
  // Use client...
</script>
```

:::warning
Never expose API keys in client-side code. Use a backend proxy for browser applications.
:::
