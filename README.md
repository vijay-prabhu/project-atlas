# Project Atlas

Smart Email Filing & Semantic Search for the AECO (Architecture, Engineering, Construction, Owner) industry.

A multi-agent AI system that automatically classifies, extracts metadata from, and files construction project emails using LangGraph, hybrid search (BM25 + vector), and RAG with citation tracking.

## Architecture

The system has three layers:

**Agent Layer** — LangGraph state graph with three specialist nodes:
- **Classifier** — determines email type (RFI, submittal, change order, transmittal, meeting minutes, etc.)
- **Extractor** — pulls structured metadata using chain-of-thought reasoning (project number, RFI number, spec sections, discipline, urgency)
- **Filer** — calls tools (project lookup, RFI matcher, sender history) and makes a confidence-scored filing decision

**Search Layer** — Hybrid retrieval combining:
- Vector similarity search via Pinecone (namespace-per-tenant isolation)
- BM25 keyword search via rank-bm25
- Reciprocal Rank Fusion for result merging
- RAG pipeline with mandatory citations and hallucination guardrails

**Infrastructure Layer** — Production-ready deployment:
- FastAPI REST API with multi-tenant middleware
- AWS SAM template (DynamoDB, SQS, Lambda, S3, EventBridge)
- TypeScript CDC processor for real-time index synchronization
- Locust load testing with multi-tenant traffic simulation

## Quick Start

```bash
# Install dependencies
python3.11 -m venv .venv && source .venv/bin/activate
make install-dev

# Start all services (DynamoDB Local + API)
./dev-start.sh

# Run tests
make test

# File an email
curl -X POST http://localhost:8000/api/v1/emails/file \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{"email": {"sender": "s.chen@pacificsteel.com", "subject": "RE: RFI-247 - Steel Connection Detail", "body": "Following up on RFI-247 regarding the steel connection detail at Grid J-7."}}'

# Search
curl -X POST http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{"query": "What is the status of RFI-247?"}'

# Stop all services
./dev-stop.sh
```

## Scale Testing

```bash
# Generate 10,000 realistic AECO emails
make generate-10k

# Load into DynamoDB Local
make generate-dynamo

# Run load tests (50 concurrent users, 2 minutes)
make loadtest-headless

# Or use the Locust web UI
make loadtest
```

The data generator produces 4,000+ emails/sec using Faker with a custom AECO provider. Emails include realistic RFIs, submittals, change orders, transmittals, and meeting minutes with proper industry terminology, spec section references, and project codes.

## Project Structure

```
src/
├── agents/          # LangGraph multi-agent system
│   ├── state.py     # TypedDict shared state
│   ├── classifier.py # Email classification (Node 1)
│   ├── extractor.py  # Metadata extraction with CoT (Node 2)
│   ├── filer.py      # Filing decision with tools (Node 3)
│   ├── graph.py      # StateGraph assembly
│   ├── checkpoints.py # HITL persistence
│   ├── guardrails.py  # Hallucination detection
│   └── search_agent.py
├── tools/           # LangChain tools
├── search/          # Vector DB + hybrid search
├── rag/             # RAG pipeline + citations
├── pipeline/        # Ingestion + CDC
├── llm/             # Multi-provider LLM client + routing
├── core/            # Config, multi-tenant, observability
└── api/             # FastAPI routes

services/            # TypeScript Lambda functions
├── cdc-processor/   # DynamoDB Streams → search index
└── webhook-receiver/ # Email webhook → SQS

infra/               # AWS SAM template
loadtest/            # Locust load tests
data/                # Sample AECO emails + versioned prompts
docs/adr/            # Architecture Decision Records
```

## Tech Stack

- **Agentic AI**: LangGraph, LangChain
- **LLMs**: OpenAI (gpt-4o, gpt-4o-mini), Anthropic (Claude), AWS Bedrock
- **Vector DB**: Pinecone (namespace-per-tenant)
- **Search**: Hybrid (BM25 + vector) with Reciprocal Rank Fusion
- **API**: FastAPI, Pydantic v2
- **AWS**: Lambda, DynamoDB, SQS, S3, EventBridge (SAM)
- **Languages**: Python 3.11, TypeScript
- **Testing**: pytest (48 tests), Locust (load testing)
