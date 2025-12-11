# AWS Infrastructure Cost Estimates

Detailed cost analysis for hosting the Memory API on AWS across different scale scenarios.

---

## Scenario 1: Startup (MVP Launch)

**Target:** 10-50 customers, ~100K API calls/month

| Service | Configuration | Monthly Cost |
|---------|---------------|--------------|
| **ECS Fargate** | 2 tasks, 1 vCPU, 2GB RAM each | $58 |
| **RDS PostgreSQL** | db.t4g.medium, 100GB, Multi-AZ | $130 |
| **ElastiCache Redis** | cache.t4g.micro, 1 node | $12 |
| **ALB** | 1 load balancer, ~10 LCU-hours | $25 |
| **S3** | 50GB storage + requests | $5 |
| **CloudWatch** | Logs, metrics, alarms | $20 |
| **Secrets Manager** | 5 secrets | $3 |
| **Route 53** | 1 hosted zone | $1 |
| **Data Transfer** | 100GB outbound | $9 |
| **ACM** | SSL certificates | Free |
| **ECR** | Container registry, 10GB | $1 |
| **Total** | | **~$264/month** |

**Notes:**
- Single region (us-east-1)
- No reserved capacity (on-demand)
- ClickHouse self-hosted on Fargate or skip for MVP

---

## Scenario 2: Growth (Product-Market Fit)

**Target:** 100-500 customers, ~1M API calls/month

| Service | Configuration | Monthly Cost |
|---------|---------------|--------------|
| **EKS** | Control plane | $73 |
| **EC2 (EKS nodes)** | 3x m6i.large (on-demand) | $234 |
| **RDS PostgreSQL** | db.r6g.large, 500GB, Multi-AZ | $438 |
| **ElastiCache Redis** | cache.r6g.large, 2 nodes | $219 |
| **ALB** | 1 load balancer, ~50 LCU-hours | $45 |
| **S3** | 200GB + requests | $15 |
| **CloudWatch** | Enhanced monitoring | $50 |
| **ClickHouse (EC2)** | 1x m6i.large, 500GB EBS | $110 |
| **Secrets Manager** | 20 secrets | $10 |
| **WAF** | Web ACL + rules | $25 |
| **Route 53** | Hosted zone + queries | $5 |
| **Data Transfer** | 500GB outbound | $45 |
| **NAT Gateway** | 1 NAT, 200GB processed | $45 |
| **ECR** | 50GB | $5 |
| **Total** | | **~$1,319/month** |

**With 1-year Reserved Instances (partial upfront):**

| Service | On-Demand | Reserved | Savings |
|---------|-----------|----------|---------|
| EC2 (EKS nodes) | $234 | $148 | 37% |
| RDS PostgreSQL | $438 | $285 | 35% |
| ElastiCache | $219 | $142 | 35% |
| **Adjusted Total** | | **~$985/month** | 25% savings |

---

## Scenario 3: Scale (Enterprise)

**Target:** 500+ customers, ~10M API calls/month

| Service | Configuration | Monthly Cost |
|---------|---------------|--------------|
| **EKS** | Control plane | $73 |
| **EC2 (EKS nodes)** | 6x m6i.xlarge (reserved) | $475 |
| **EC2 Spot (burst)** | 3x m6i.large avg | $70 |
| **RDS PostgreSQL** | db.r6g.xlarge, 1TB, Multi-AZ, read replica | $1,150 |
| **ElastiCache Redis** | cache.r6g.xlarge, 3-node cluster | $548 |
| **ALB** | 2 load balancers, ~200 LCU-hours | $120 |
| **S3** | 1TB + requests | $50 |
| **CloudWatch** | Full observability | $150 |
| **ClickHouse (EC2)** | 2x m6i.xlarge, 1TB EBS | $350 |
| **OpenSearch** | 3x m6g.large (logs/search) | $330 |
| **Secrets Manager** | 50 secrets | $20 |
| **WAF** | Full rule set | $50 |
| **Shield Advanced** | DDoS protection | $3,000 |
| **Route 53** | Multiple zones + health checks | $20 |
| **Data Transfer** | 2TB outbound | $180 |
| **NAT Gateway** | 2 NATs, 1TB processed | $135 |
| **Backup** | AWS Backup | $50 |
| **ECR** | 100GB | $10 |
| **Total** | | **~$6,781/month** |

**Without Shield Advanced:** ~$3,781/month

---

## Upstream API Costs (OpenAI)

These costs are **in addition** to AWS infrastructure:

| Usage Level | Monthly Tokens | Embedding Calls | Est. OpenAI Cost |
|-------------|----------------|-----------------|------------------|
| Startup | 10M tokens | 100K | ~$50 |
| Growth | 100M tokens | 1M | ~$400 |
| Scale | 1B tokens | 10M | ~$3,500 |

**Model Pricing Used:**
- gpt-4o-mini: $0.15/$0.60 per 1M tokens (input/output)
- text-embedding-3-small: $0.02 per 1M tokens
- Assumes 70% prefiltering savings from benchmark learnings

---

## Total Cost Summary

| Scenario | AWS Infra | OpenAI API | Total | Per Customer* |
|----------|-----------|------------|-------|---------------|
| **Startup** | $264 | $50 | **$314** | $6-31 |
| **Growth** | $985 | $400 | **$1,385** | $3-14 |
| **Scale** | $3,781 | $3,500 | **$7,281** | $7-15 |

*Per customer assumes 10-50 (startup), 100-500 (growth), 500+ (scale)

---

## Cost Optimization Strategies

### Immediate Savings (10-30%)

| Strategy | Savings | Effort |
|----------|---------|--------|
| Reserved Instances (1-yr) | 30-40% on compute | Low |
| Savings Plans | 20-30% on Fargate/Lambda | Low |
| S3 Intelligent-Tiering | 20-40% on storage | Low |
| Right-sizing | 10-20% | Medium |

### Medium-term Savings (30-50%)

| Strategy | Savings | Effort |
|----------|---------|--------|
| Spot Instances for workers | 60-70% on batch compute | Medium |
| Reserved Instances (3-yr) | 50-60% on compute | Medium |
| Graviton (ARM) instances | 20% better price/perf | Medium |
| Aurora Serverless v2 | Variable, can reduce 30% | Medium |

### Architecture Optimizations

| Optimization | Impact |
|--------------|--------|
| **Response caching (Redis)** | 30-50% reduction in LLM calls |
| **Batch API for bulk ops** | 50% reduction in OpenAI costs |
| **Edge caching (CloudFront)** | Reduce origin requests |
| **Connection pooling** | Reduce RDS costs |

---

## Recommended Starting Configuration

For MVP launch, I recommend starting lean:

```
┌─────────────────────────────────────────────────────────────┐
│                    Minimal Viable Stack                      │
│                        ~$200/month                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│   │   Fargate   │────▶│  RDS Postgres│     │    Redis    │   │
│   │   2 tasks   │     │  t4g.medium │     │  t4g.micro  │   │
│   │   $58/mo    │     │   $95/mo    │     │   $12/mo    │   │
│   └─────────────┘     └─────────────┘     └─────────────┘   │
│          │                                                   │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │     ALB     │      Skip for MVP:                        │
│   │   $22/mo    │      - ClickHouse (use RDS)               │
│   └─────────────┘      - WAF (use Cloudflare free)          │
│                        - Multi-region                        │
│                        - Read replicas                       │
└─────────────────────────────────────────────────────────────┘
```

**Scaling triggers:**
- Move to EKS when: >5 services or need advanced orchestration
- Add read replica when: DB CPU > 70%
- Add ClickHouse when: >1M events/day analytics needed
- Add WAF when: experiencing attacks or compliance required

---

## Monthly Spend Projection (Year 1)

| Month | Customers | API Calls | AWS | OpenAI | Total |
|-------|-----------|-----------|-----|--------|-------|
| 1-3 | 10 | 50K | $264 | $25 | $289 |
| 4-6 | 50 | 250K | $400 | $100 | $500 |
| 7-9 | 150 | 750K | $700 | $300 | $1,000 |
| 10-12 | 300 | 1.5M | $985 | $500 | $1,485 |

**Year 1 Total:** ~$9,800

---

## Break-even Analysis

| Plan | Price | Customers Needed (Startup Costs) |
|------|-------|----------------------------------|
| Free | $0 | N/A |
| Starter | $49/mo | 7 customers |
| Professional | $299/mo | 2 customers |
| Enterprise | $1000+/mo | 1 customer |

**Target:** 5 Starter + 2 Pro = $843/mo revenue = profitable at Startup tier

---

## Alternative: Serverless Architecture

If traffic is unpredictable, consider serverless:

| Service | Configuration | Cost Model |
|---------|---------------|------------|
| **Lambda** | API handlers | $0.20 per 1M requests |
| **API Gateway** | HTTP API | $1.00 per 1M requests |
| **Aurora Serverless v2** | PostgreSQL | $0.12 per ACU-hour |
| **ElastiCache Serverless** | Redis | $0.125 per ECPU-hour |
| **DynamoDB** | Usage tracking | $1.25 per 1M writes |

**Estimated at 100K requests/month:** ~$150-200/month

Pros: Scales to zero, no idle costs
Cons: Cold starts, vendor lock-in, harder to debug
