# Memory API Service Plan

## Executive Summary

This document outlines the plan for deploying the Memory System as a commercial API service for external customers. Based on the HaluMem-long benchmark results (57.2% update accuracy, 58.1% QA correctness, 0% update hallucination), the system demonstrates production-ready capabilities for memory storage and retrieval with room for improvement in dynamic updates and QA hallucination reduction.

---

## Table of Contents

1. [API Architecture](#1-api-architecture)
2. [Security Design](#2-security-design)
3. [Billing & Usage System](#3-billing--usage-system)
4. [Internal Cost Analytics](#4-internal-cost-analytics)
5. [Customer Documentation](#5-customer-documentation)
6. [Implementation Roadmap](#6-implementation-roadmap)

---

## 1. API Architecture

### 1.1 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **API Framework** | FastAPI (Python) | Async support, auto-docs, Pydantic validation |
| **Database** | PostgreSQL + pgvector | Production-grade, vector similarity support |
| **Cache** | Redis | Rate limiting, session cache, hot data |
| **Queue** | Celery + Redis | Async job processing for batch operations |
| **Load Balancer** | nginx / AWS ALB | SSL termination, routing |
| **Container** | Docker + Kubernetes | Scalability, orchestration |

### 1.2 API Endpoints

#### Core Memory Operations

```
POST   /v1/memories                    # Create memory
GET    /v1/memories/{memory_id}        # Get single memory
GET    /v1/memories                    # List memories (paginated)
PUT    /v1/memories/{memory_id}        # Update memory
DELETE /v1/memories/{memory_id}        # Delete memory
DELETE /v1/memories                    # Clear all memories (with confirmation)
```

#### Retrieval Operations

```
POST   /v1/query                       # Query with context, get relevant memories
POST   /v1/query/answer                # Query and generate answer
POST   /v1/retrieve                    # Retrieve relevant memories without answering
```

#### Batch Operations

```
POST   /v1/batch/memories              # Bulk create memories
POST   /v1/batch/extract               # Extract memories from dialogue
POST   /v1/batch/embeddings            # Generate embeddings for memories
GET    /v1/batch/{job_id}              # Get batch job status
```

#### Account & Usage

```
GET    /v1/usage                       # Get current usage stats
GET    /v1/usage/history               # Historical usage data
GET    /v1/account                     # Account details and limits
GET    /v1/account/api-keys            # List API keys
POST   /v1/account/api-keys            # Create new API key
DELETE /v1/account/api-keys/{key_id}   # Revoke API key
```

### 1.3 Request/Response Format

**Standard Request Headers:**
```http
Authorization: Bearer <api_key>
X-Request-ID: <uuid>                   # Optional, for tracing
Content-Type: application/json
```

**Standard Response Format:**
```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "request_id": "uuid",
    "processing_time_ms": 45,
    "usage": {
      "tokens_processed": 1250,
      "memories_accessed": 15,
      "model_calls": 2
    }
  }
}
```

**Error Response Format:**
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit of 100 requests/minute exceeded",
    "details": {
      "retry_after": 30,
      "limit": 100,
      "window": "minute"
    }
  },
  "meta": {
    "request_id": "uuid"
  }
}
```

### 1.4 API Versioning Strategy

- URL-based versioning: `/v1/`, `/v2/`
- Deprecation policy: 12 months notice before sunsetting
- Breaking changes only in major versions
- Non-breaking additions within versions

---

## 2. Security Design

### 2.1 Authentication & Authorization

#### API Key Management

```
API Key Format: mem_live_<32-char-random>  (production)
               mem_test_<32-char-random>  (sandbox)
```

| Feature | Implementation |
|---------|---------------|
| **Key Storage** | Hashed with bcrypt, only prefix shown in dashboard |
| **Key Rotation** | Self-service rotation with grace period |
| **Key Scopes** | read, write, delete, admin |
| **Key Expiry** | Optional expiration dates |

#### Authentication Flow

```
┌─────────┐      ┌─────────────┐      ┌─────────────┐
│ Client  │──────│ API Gateway │──────│ Auth Service│
└─────────┘      └─────────────┘      └─────────────┘
     │                  │                    │
     │ API Key in       │ Validate key       │
     │ Authorization    │────────────────────│
     │ header           │                    │
     │                  │  Return: org_id,   │
     │                  │  scopes, limits    │
     │                  │◀───────────────────│
     │                  │                    │
     │                  │ Add context to     │
     │                  │ request headers    │
     │                  ▼                    │
     │           ┌─────────────┐             │
     │           │ API Service │             │
     │           └─────────────┘             │
```

### 2.2 Rate Limiting

#### Tiered Rate Limits

| Tier | Requests/min | Requests/day | Concurrent |
|------|--------------|--------------|------------|
| Free | 10 | 100 | 2 |
| Starter | 60 | 5,000 | 5 |
| Professional | 300 | 50,000 | 20 |
| Enterprise | Custom | Custom | Custom |

#### Implementation

```python
# Redis-based sliding window rate limiting
RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, window)
end
return current <= limit
"""
```

### 2.3 Data Isolation

- **Tenant Isolation**: Each customer's memories stored with `org_id` prefix
- **Database**: Row-level security in PostgreSQL
- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Memory Isolation**: No cross-tenant data access possible

### 2.4 Audit Logging

All API calls logged with:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "org_id": "org_abc123",
  "api_key_id": "key_xyz",
  "endpoint": "POST /v1/memories",
  "ip_address": "192.168.1.1",
  "user_agent": "MemorySDK/1.0",
  "request_id": "req_uuid",
  "response_code": 201,
  "processing_time_ms": 45,
  "tokens_used": 150
}
```

### 2.5 Security Checklist

- [ ] HTTPS only (HSTS enabled)
- [ ] Input validation (Pydantic models)
- [ ] SQL injection prevention (parameterized queries)
- [ ] Request size limits (10MB max)
- [ ] API key in header only (never in URL)
- [ ] CORS configuration (whitelist origins)
- [ ] DDoS protection (Cloudflare/AWS Shield)
- [ ] Regular security audits

---

## 3. Billing & Usage System

### 3.1 Pricing Model

#### Usage-Based Pricing

| Resource | Unit | Free | Starter | Pro | Enterprise |
|----------|------|------|---------|-----|------------|
| **Memory Storage** | per 1K memories/month | 100 | 10,000 | 100,000 | Unlimited |
| **API Calls** | per 1K calls | 100 | 10,000 | 100,000 | Custom |
| **Tokens Processed** | per 1M tokens | 10K | 1M | 20M | Custom |
| **Embeddings** | per 1K embeddings | 100 | 10,000 | 100,000 | Custom |
| **Base Price** | per month | $0 | $49 | $299 | Custom |
| **Overage - Calls** | per 1K | - | $0.50 | $0.30 | Custom |
| **Overage - Tokens** | per 1M | - | $5.00 | $3.00 | Custom |

### 3.2 Usage Tracking Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Usage Collector                              │
│  - Intercepts all requests                                       │
│  - Extracts: tokens, memories accessed, model calls              │
│  - Writes to Redis (real-time) + Kafka (durability)             │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│    Redis (Real-time)     │     │      Kafka → ClickHouse         │
│  - Current period usage  │     │  - Historical usage data        │
│  - Rate limit counters   │     │  - Analytics and reporting      │
│  - Quick quota checks    │     │  - Billing aggregation          │
└─────────────────────────┘     └─────────────────────────────────┘
                                              │
                                              ▼
                              ┌─────────────────────────────────┐
                              │       Billing Aggregator         │
                              │  - Daily/monthly rollups         │
                              │  - Invoice generation            │
                              │  - Stripe integration            │
                              └─────────────────────────────────┘
```

### 3.3 Usage Data Schema

```sql
CREATE TABLE usage_events (
    id UUID PRIMARY KEY,
    org_id VARCHAR(50) NOT NULL,
    api_key_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    endpoint VARCHAR(100) NOT NULL,

    -- Usage metrics
    tokens_input INT DEFAULT 0,
    tokens_output INT DEFAULT 0,
    memories_read INT DEFAULT 0,
    memories_written INT DEFAULT 0,
    embeddings_generated INT DEFAULT 0,
    model_calls INT DEFAULT 0,

    -- Cost tracking (internal)
    cost_usd DECIMAL(10, 6) DEFAULT 0,

    -- Metadata
    request_id UUID,
    processing_time_ms INT,

    INDEX idx_org_timestamp (org_id, timestamp),
    INDEX idx_billing_period (org_id, DATE_TRUNC('month', timestamp))
);

CREATE TABLE usage_daily_rollup (
    org_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,

    total_requests INT DEFAULT 0,
    total_tokens_input BIGINT DEFAULT 0,
    total_tokens_output BIGINT DEFAULT 0,
    total_memories_read BIGINT DEFAULT 0,
    total_memories_written BIGINT DEFAULT 0,
    total_embeddings BIGINT DEFAULT 0,
    total_model_calls BIGINT DEFAULT 0,
    total_cost_usd DECIMAL(12, 4) DEFAULT 0,

    PRIMARY KEY (org_id, date)
);
```

### 3.4 Billing Integration (Stripe)

```python
# Billing flow
class BillingService:

    async def sync_usage_to_stripe(self, org_id: str, period_end: date):
        """Report usage to Stripe for metered billing."""
        usage = await self.get_period_usage(org_id, period_end)

        # Report each usage type as a separate meter
        await stripe.billing.MeterEvent.create(
            event_name="api_calls",
            payload={
                "value": usage.total_requests,
                "stripe_customer_id": org.stripe_customer_id
            }
        )
        await stripe.billing.MeterEvent.create(
            event_name="tokens_processed",
            payload={
                "value": usage.total_tokens,
                "stripe_customer_id": org.stripe_customer_id
            }
        )

    async def check_quota(self, org_id: str) -> QuotaStatus:
        """Real-time quota check before processing request."""
        current = await redis.hgetall(f"usage:{org_id}:current")
        limits = await self.get_org_limits(org_id)

        return QuotaStatus(
            requests_remaining=limits.requests - current.requests,
            tokens_remaining=limits.tokens - current.tokens,
            can_proceed=current.requests < limits.requests
        )
```

### 3.5 Customer Usage Dashboard

**Dashboard Features:**
- Real-time usage graphs (requests, tokens, memories)
- Current billing period summary
- Projected month-end costs
- Usage by endpoint breakdown
- API key usage comparison
- Export usage data (CSV, JSON)
- Usage alerts configuration

**Alert Configuration:**
```json
{
  "alerts": [
    {
      "metric": "monthly_spend",
      "threshold": 500,
      "threshold_type": "absolute",
      "notify": ["email", "webhook"]
    },
    {
      "metric": "api_calls",
      "threshold": 80,
      "threshold_type": "percent_of_limit",
      "notify": ["email"]
    }
  ]
}
```

---

## 4. Internal Cost Analytics

### 4.1 Cost Tracking Requirements

Track costs from all upstream providers and infrastructure:

| Cost Category | Source | Tracking Method |
|---------------|--------|-----------------|
| **LLM API Calls** | OpenAI, Anthropic | Per-request token counting |
| **Embeddings** | OpenAI | Per-request dimension counting |
| **Database** | PostgreSQL/RDS | Monthly infrastructure cost allocation |
| **Compute** | Kubernetes/EC2 | Resource usage per request |
| **Bandwidth** | CDN/Transfer | Bytes transferred |
| **Storage** | S3/EBS | GB-months allocated per org |

### 4.2 Cost Attribution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Cost Attribution Service                     │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   LLM Costs      │  │ Infrastructure  │  │  Storage Costs  │
│                  │  │     Costs       │  │                 │
│ • Token pricing  │  │ • CPU hours     │  │ • Memory count  │
│ • Model tiers    │  │ • Memory GB-hrs │  │ • Embedding dim │
│ • Batch vs sync  │  │ • Network       │  │ • Backups       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  Cost Database  │
                    │  (ClickHouse)   │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Dashboards    │
                    │   (Grafana)     │
                    └─────────────────┘
```

### 4.3 LLM Cost Tracking

Based on HaluMem benchmark learnings (98% cost reduction with prefiltering):

```python
class LLMCostTracker:
    # Current OpenAI pricing (update regularly)
    PRICING = {
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4.1-nano": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
        "text-embedding-3-small": {"input": 0.02 / 1_000_000},
        "text-embedding-3-large": {"input": 0.13 / 1_000_000},
    }

    def calculate_request_cost(self, request_log: RequestLog) -> CostBreakdown:
        """Calculate total cost for a single API request."""
        costs = {}

        for model_call in request_log.model_calls:
            model = model_call.model
            pricing = self.PRICING[model]

            input_cost = model_call.input_tokens * pricing["input"]
            output_cost = model_call.output_tokens * pricing.get("output", 0)

            costs[model] = costs.get(model, 0) + input_cost + output_cost

        return CostBreakdown(
            total=sum(costs.values()),
            by_model=costs,
            margin=self.calculate_margin(request_log.org_id, sum(costs.values()))
        )
```

### 4.4 Cost Analytics Dashboard (Internal)

**Metrics to Track:**

1. **Gross Margin by Customer**
   - Revenue vs. cost per organization
   - Identify unprofitable accounts
   - Cost trends over time

2. **Cost per Operation**
   - Average cost per API call by endpoint
   - Cost per 1K tokens by operation type
   - Embedding cost per memory

3. **Model Efficiency**
   - Cost comparison: prefilter + nano model vs. full model
   - Cache hit rates and cost savings
   - Batch vs. sync cost efficiency

4. **Infrastructure Utilization**
   - Cost per request by pod/instance
   - Database query costs
   - Storage growth rate

**Example Dashboard Queries:**

```sql
-- Gross margin by customer (monthly)
SELECT
    org_id,
    SUM(revenue_usd) as revenue,
    SUM(cost_usd) as cost,
    SUM(revenue_usd) - SUM(cost_usd) as gross_profit,
    (SUM(revenue_usd) - SUM(cost_usd)) / SUM(revenue_usd) * 100 as margin_pct
FROM billing_summary
WHERE month = '2024-01'
GROUP BY org_id
ORDER BY gross_profit DESC;

-- Cost breakdown by operation type
SELECT
    endpoint,
    COUNT(*) as requests,
    AVG(cost_usd) as avg_cost,
    SUM(cost_usd) as total_cost,
    AVG(tokens_input + tokens_output) as avg_tokens
FROM usage_events
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY endpoint
ORDER BY total_cost DESC;
```

### 4.5 Cost Optimization Strategies

Based on HaluMem benchmark findings:

| Strategy | Expected Savings | Implementation |
|----------|-----------------|----------------|
| **Embedding Prefilter** | 98% on retrieval | Already implemented |
| **Nano Model for Scoring** | 90% vs full model | Already implemented |
| **Response Caching** | 30-50% on repeated queries | Redis cache layer |
| **Batch API Usage** | 50% on bulk operations | OpenAI Batch API |
| **Smart Model Routing** | 20-40% | Use smaller models for simple queries |

---

## 5. Customer Documentation

### 5.1 Documentation Structure

```
docs/
├── index.md                    # Landing page
├── getting-started/
│   ├── quickstart.md           # 5-minute guide
│   ├── authentication.md       # API keys setup
│   ├── first-request.md        # Hello world
│   └── concepts.md             # Core concepts
├── api-reference/
│   ├── overview.md             # API conventions
│   ├── memories.md             # Memory CRUD
│   ├── retrieval.md            # Query endpoints
│   ├── batch.md                # Batch operations
│   └── errors.md               # Error codes
├── guides/
│   ├── memory-extraction.md    # Extracting from conversations
│   ├── retrieval-tuning.md     # Optimizing retrieval
│   ├── handling-updates.md     # Memory updates
│   └── best-practices.md       # Production tips
├── sdks/
│   ├── python.md               # Python SDK
│   ├── javascript.md           # JavaScript SDK
│   └── rest.md                 # Raw REST usage
├── dashboard/
│   ├── usage.md                # Understanding usage
│   ├── billing.md              # Billing and invoices
│   └── api-keys.md             # Key management
└── changelog.md                # Version history
```

### 5.2 API Reference Format (OpenAPI)

```yaml
openapi: 3.1.0
info:
  title: Memory API
  version: 1.0.0
  description: |
    The Memory API provides persistent, intelligent memory storage
    and retrieval for AI applications.

servers:
  - url: https://api.memory.example.com/v1
    description: Production

security:
  - bearerAuth: []

paths:
  /memories:
    post:
      summary: Create a memory
      operationId: createMemory
      tags: [Memories]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateMemoryRequest'
            example:
              text: "User prefers dark mode interfaces"
              metadata:
                category: "preference"
                source: "onboarding"
      responses:
        '201':
          description: Memory created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MemoryResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '429':
          $ref: '#/components/responses/RateLimited'

  /query:
    post:
      summary: Query memories
      operationId: queryMemories
      tags: [Retrieval]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/QueryRequest'
            example:
              context: "User is asking about their UI preferences"
              max_memories: 10
              relevance_threshold: 0.5
      responses:
        '200':
          description: Query results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QueryResponse'
```

### 5.3 Interactive Documentation

**Tools:**
- **Swagger UI** - Interactive API explorer
- **Redoc** - Beautiful reference documentation
- **Postman Collection** - Pre-built request examples

**Features:**
- Try endpoints directly in browser
- Copy-paste code snippets (curl, Python, JS)
- Authentication helper
- Response schema viewer

### 5.4 SDK Design

**Python SDK Example:**

```python
from memory_api import MemoryClient

# Initialize
client = MemoryClient(api_key="mem_live_xxx")

# Create memories
memory = client.memories.create(
    text="User works as a software engineer at TechCorp",
    metadata={"category": "professional", "confidence": 0.95}
)

# Query with context
results = client.query(
    context="What does the user do for work?",
    max_memories=5
)

for memory in results.memories:
    print(f"[{memory.relevance:.2f}] {memory.text}")

# Get answer directly
answer = client.query_with_answer(
    context="What industry does the user work in?",
    prompt_template="Based on the memories, answer: {question}"
)
print(answer.response)
```

**JavaScript SDK Example:**

```javascript
import { MemoryClient } from '@memory-api/sdk';

const client = new MemoryClient({ apiKey: 'mem_live_xxx' });

// Create memory
const memory = await client.memories.create({
  text: 'User prefers TypeScript over JavaScript',
  metadata: { category: 'preference' }
});

// Query
const results = await client.query({
  context: 'What programming languages does the user like?',
  maxMemories: 10
});

results.memories.forEach(m => {
  console.log(`[${m.relevance.toFixed(2)}] ${m.text}`);
});
```

### 5.5 Documentation Hosting

| Option | Pros | Cons |
|--------|------|------|
| **GitBook** | Beautiful, easy editing | Cost at scale |
| **Docusaurus** | Free, React-based, versioning | More setup |
| **ReadMe.io** | Interactive, metrics | Expensive |
| **Mintlify** | Modern, fast | Newer platform |

**Recommendation:** Docusaurus for cost-effectiveness and flexibility, with OpenAPI integration for API reference.

---

## 6. Implementation Roadmap

### Phase 1: Core API (Weeks 1-4)

```
Week 1-2: API Foundation
├── FastAPI project setup
├── PostgreSQL + pgvector migration
├── Core endpoints (CRUD memories)
├── Basic authentication (API keys)
└── Docker containerization

Week 3-4: Retrieval & Security
├── Query endpoints
├── Rate limiting (Redis)
├── Input validation
├── Error handling
├── Basic logging
└── Integration tests
```

**Deliverables:**
- [ ] Working API with authentication
- [ ] Memory CRUD operations
- [ ] Query and retrieval endpoints
- [ ] Rate limiting
- [ ] Docker deployment

### Phase 2: Billing & Usage (Weeks 5-8)

```
Week 5-6: Usage Tracking
├── Usage event collection
├── Redis real-time counters
├── ClickHouse historical storage
├── Daily rollup jobs
└── Quota enforcement

Week 7-8: Billing Integration
├── Stripe integration
├── Customer portal
├── Usage dashboard (basic)
├── Invoice generation
└── Payment webhooks
```

**Deliverables:**
- [ ] Real-time usage tracking
- [ ] Stripe billing integration
- [ ] Customer usage dashboard
- [ ] Automated invoicing

### Phase 3: Analytics & Optimization (Weeks 9-12)

```
Week 9-10: Internal Analytics
├── Cost attribution pipeline
├── Margin tracking
├── Grafana dashboards
├── Alerting (PagerDuty)
└── Cost optimization recommendations

Week 11-12: Customer Analytics
├── Enhanced usage dashboard
├── Usage projections
├── Cost alerts
├── Export capabilities
└── API for usage data
```

**Deliverables:**
- [ ] Internal cost dashboards
- [ ] Margin analysis by customer
- [ ] Customer usage projections
- [ ] Usage alerts

### Phase 4: Documentation & SDKs (Weeks 13-16)

```
Week 13-14: Documentation
├── Docusaurus setup
├── API reference (OpenAPI)
├── Getting started guides
├── Integration guides
└── Interactive examples

Week 15-16: SDKs & Polish
├── Python SDK
├── JavaScript SDK
├── Postman collection
├── Changelog
└── Support documentation
```

**Deliverables:**
- [ ] Complete documentation site
- [ ] Python SDK (PyPI)
- [ ] JavaScript SDK (npm)
- [ ] Postman collection

---

## 7. Infrastructure Diagram

```
                                    ┌─────────────────┐
                                    │   Cloudflare    │
                                    │   (DDoS/WAF)    │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │   Load Balancer │
                                    │   (nginx/ALB)   │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
           │   API Pod 1     │     │   API Pod 2     │     │   API Pod N     │
           │   (FastAPI)     │     │   (FastAPI)     │     │   (FastAPI)     │
           └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                    │                        │                        │
                    └────────────────────────┼────────────────────────┘
                                             │
         ┌───────────────┬───────────────────┼───────────────┬───────────────┐
         │               │                   │               │               │
┌────────▼────────┐ ┌────▼─────┐    ┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼──────┐
│   PostgreSQL    │ │  Redis   │    │    Celery     │ │ClickHouse │ │   OpenAI     │
│   + pgvector    │ │ (cache/  │    │   Workers     │ │(analytics)│ │    API       │
│                 │ │  limits) │    │               │ │           │ │              │
└─────────────────┘ └──────────┘    └───────────────┘ └───────────┘ └──────────────┘
```

---

## 8. Cost Estimates

### Infrastructure Costs (Monthly)

| Component | Specification | Estimated Cost |
|-----------|---------------|----------------|
| Kubernetes (EKS) | 3x m5.large | $200 |
| PostgreSQL (RDS) | db.r5.large | $150 |
| Redis (ElastiCache) | cache.r5.large | $100 |
| ClickHouse | Self-hosted on k8s | $50 |
| Load Balancer | ALB | $30 |
| Storage (S3/EBS) | 500GB | $50 |
| Monitoring (Datadog) | - | $100 |
| **Total Infrastructure** | | **~$680/month** |

### Margin Analysis

Based on benchmark costs ($65 for 20M tokens with prefiltering):

| Scenario | Your Cost | Customer Pays | Margin |
|----------|-----------|---------------|--------|
| 1M tokens processed | $3.25 | $15 | 78% |
| 10K memories stored | $0.50 | $5 | 90% |
| 1K API calls | $0.15 | $0.50 | 70% |

**Target Gross Margin: 70-80%**

---

## 9. Success Metrics

### Technical KPIs

- API uptime: 99.9%
- P95 latency: < 500ms
- Error rate: < 0.1%

### Business KPIs

- Customer acquisition cost (CAC)
- Customer lifetime value (LTV)
- Monthly recurring revenue (MRR)
- Gross margin per customer
- Churn rate

### Quality KPIs (from benchmarks)

- Update accuracy: Target 70% (current 57.2%)
- QA correctness: Target 70% (current 58.1%)
- QA hallucination rate: Target 30% (current 56.9%)

---

## Appendix A: Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_API_KEY` | 401 | API key is invalid or expired |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `QUOTA_EXCEEDED` | 402 | Usage quota exceeded |
| `MEMORY_NOT_FOUND` | 404 | Memory ID not found |
| `INVALID_REQUEST` | 400 | Request validation failed |
| `INTERNAL_ERROR` | 500 | Internal server error |

## Appendix B: Webhook Events

```json
{
  "event": "usage.threshold_reached",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "org_id": "org_abc123",
    "metric": "api_calls",
    "current_value": 8500,
    "threshold": 8000,
    "threshold_percent": 80,
    "limit": 10000
  }
}
```

Events:
- `usage.threshold_reached` - Usage approaching limit
- `usage.quota_exceeded` - Quota exceeded
- `billing.payment_failed` - Payment failed
- `billing.invoice_created` - New invoice ready
