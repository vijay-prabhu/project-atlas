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
pip install streamlit faker locust

# Start all services (DynamoDB Local + API + Dashboard)
./dev-start.sh

# Run tests
make test

# Stop all services
./dev-stop.sh
```

After `./dev-start.sh`, open:
- **http://localhost:8501** — Demo Dashboard (file emails, search, batch demo, agent trace viewer, HITL reviews)
- **http://localhost:8000/docs** — Swagger API docs
- **http://localhost:8002** — DynamoDB Admin UI

### Demo Dashboard

The Streamlit dashboard has 5 pages:

| Page | What it does |
|------|-------------|
| **File Email** | Submit an email (or use quick-fill samples), watch classification, metadata extraction, and filing decision with confidence scoring |
| **Search** | Natural language search with hybrid retrieval, RAG answers, citations, and guardrail warnings |
| **Batch Demo** | Generate and file 5-50 random AECO emails, see classification accuracy and distribution charts |
| **Agent Trace Viewer** | Generate a random email, visualize the full pipeline flow (classify → extract → file → result) with timing |
| **Pending Reviews** | Human-in-the-loop interface — approve, reject, or correct emails that need human review |

## Scale Testing

```bash
# Generate 10,000 realistic AECO emails
make generate-10k

# Generate 100K and load into DynamoDB Local
make generate-100k
make generate-dynamo

# Run load tests — Locust web UI at http://localhost:8089
make loadtest

# Or headless mode (50 concurrent users, 2 minutes)
make loadtest-headless
```

The data generator produces **4,375 emails/sec** using Faker with a custom AECO provider (20 companies, 10 projects, 14 spec sections, 15 RFI templates). DynamoDB bulk loading runs at **3,236 records/sec** via `batch_writer()`.

### Load Test Results (50 concurrent users, multi-tenant)

| Endpoint | Requests | Failures | P50 (ms) | P95 (ms) | Req/s |
|----------|----------|----------|----------|----------|-------|
| POST /emails/file | 977 | 0 | 20 | 50 | 16.8 |
| POST /search | 582 | 0 | 8 | 46 | 10.0 |
| POST /feedback | 171 | 0 | 5 | 42 | 2.9 |
| GET /health | 227 | 0 | 3 | 22 | 3.9 |
| **Total** | **1,957** | **0** | **15** | **48** | **33.7** |

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

ui/                  # Streamlit demo dashboard
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
- **UI**: Streamlit (demo dashboard)
- **Testing**: pytest (48 tests), Locust (load testing)
