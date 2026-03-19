# ADR-001: Vector Database Selection

## Status
Accepted

## Context
Project Atlas needs a vector database to store and search embeddings across billions of AECO project documents. We need namespace-per-tenant isolation, HNSW indexing for fast approximate nearest neighbor search, and the ability to handle high-volume upserts from the CDC pipeline.

## Options Considered

### Pinecone (Serverless)
- Managed service, no infrastructure to run
- Native namespace support for tenant isolation
- HNSW indexing with configurable parameters
- Free tier: 1 index, 100K vectors
- Scales automatically

### pgvector (PostgreSQL extension)
- Runs inside existing PostgreSQL
- HNSW indexing (added in pgvector 0.5.0)
- No separate service to manage
- Row-level security for tenant isolation
- Needs manual capacity planning

### OpenSearch (k-NN plugin)
- Full-text search + vector search in one system
- Native BM25 for keyword search
- Index-per-tenant isolation model
- More operational overhead
- AWS managed option available

## Decision
Use **Pinecone** for the vector store with **rank-bm25** for keyword search.

## Why
1. Namespace-per-tenant gives us physical isolation without managing separate indexes
2. Serverless scaling means we don't manage infrastructure as we go from demo to production
3. Free tier is sufficient for portfolio/development
4. Separating vector search (Pinecone) from keyword search (BM25) keeps each system simple

## HNSW Tuning
- `m = 16` — connections per node. Higher = better recall, more memory
- `ef_construction = 256` — build-time search width. Higher = better index quality, slower builds
- `metric = cosine` — matches our embedding model (text-embedding-3-small)

These parameters were chosen because:
- m=16 is the sweet spot for 1536-dimension embeddings (OpenAI)
- ef_construction=256 gives good recall without excessive build time
- We measured IVFFlat failing on small datasets (ContractLens migration 003) — HNSW works at any scale

## Consequences
- Need to manage two search systems (Pinecone + BM25) instead of one
- Pinecone free tier limits to 100K vectors — need paid plan for production
- No SQL joins between vector results and relational data
