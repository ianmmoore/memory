# Memory API

A secure, scalable API service for intelligent memory storage and retrieval.

## Features

- **Memory Storage**: Store, update, and delete memories with metadata
- **Semantic Retrieval**: AI-powered memory search using relevance scoring
- **Answer Generation**: Generate answers based on stored memories
- **Multi-tenant**: Secure isolation between organizations
- **Usage Tracking**: Real-time usage monitoring and billing
- **Rate Limiting**: Per-organization rate limits by plan tier

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone and start services
cd memory_api
docker-compose up -d

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/memory_api"
export REDIS_URL="redis://localhost:6379/0"
export OPENAI_API_KEY="your-key"
export SECRET_KEY="your-secret-key"

# Run server
python -m uvicorn api.main:app --reload
```

## API Usage

### Create Account

```bash
curl -X POST http://localhost:8000/v1/signup \
  -H "Content-Type: application/json" \
  -d '{"name": "My Company", "email": "me@example.com"}'
```

### Store Memory

```bash
curl -X POST http://localhost:8000/v1/memories \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers dark mode"}'
```

### Query Memories

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"context": "What are the users UI preferences?"}'
```

## Project Structure

```
memory_api/
├── api/                    # FastAPI application
│   ├── routes/             # API endpoints
│   ├── middleware/         # Auth, rate limiting, usage
│   └── models/             # Pydantic schemas
├── billing/                # Pricing and usage
├── config/                 # Settings and pricing config
├── db/                     # Database models
├── docs/                   # Documentation
│   ├── internal/           # Architecture docs
│   └── customer/           # API docs
└── tests/                  # Test suite
```

## Configuration

Environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection | Yes |
| `REDIS_URL` | Redis connection | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `SECRET_KEY` | App secret | Yes |
| `STRIPE_API_KEY` | Stripe API key | No |

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=api --cov=billing

# Specific tests
pytest tests/test_memories.py -v
```

## Documentation

- **Customer Docs**: [docs/customer/](docs/customer/)
- **Architecture**: [docs/internal/ARCHITECTURE.md](docs/internal/ARCHITECTURE.md)
- **API Reference**: [docs/customer/API_REFERENCE.md](docs/customer/API_REFERENCE.md)
- **Interactive Docs**: http://localhost:8000/docs

## License

Proprietary - All rights reserved.
