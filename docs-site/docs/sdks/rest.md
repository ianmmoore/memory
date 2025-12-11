---
sidebar_position: 3
---

# REST API

Use the API directly without an SDK.

## Base URL

```
https://api.memory-api.com/v1
```

## Authentication

```bash
curl https://api.memory-api.com/v1/memories \
  -H "Authorization: Bearer mem_live_your_api_key"
```

## Create Memory

```bash
curl -X POST https://api.memory-api.com/v1/memories \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Memory content", "metadata": {"key": "value"}}'
```

## Query Memories

```bash
curl -X POST https://api.memory-api.com/v1/query \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"context": "Your question here"}'
```

## Postman Collection

Import our Postman collection for easy testing:

[![Run in Postman](https://run.pstmn.io/button.svg)](https://app.getpostman.com/run-collection/xxx)

## OpenAPI Spec

Download the OpenAPI specification:

```bash
curl https://api.memory-api.com/openapi.json -o openapi.json
```

Use with any OpenAPI-compatible tool.
