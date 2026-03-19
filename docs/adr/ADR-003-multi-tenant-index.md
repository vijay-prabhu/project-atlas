# ADR-003: Multi-Tenant Data Isolation at the Index Level

## Status
Accepted

## Context
Project Atlas serves multiple AECO firms. Data isolation is non-negotiable — Firm A must never see Firm B's project emails through search.

## Decision
Use **namespace-per-tenant** isolation in Pinecone, enforced at three layers:

### Layer 1: Ingestion
Every document is tagged with `tenant_id` at ingestion time. The CDC pipeline validates tenant context before indexing.

### Layer 2: Query
Tenant namespace is injected into every search query. This is not optional — it's enforced by the `TenantMiddleware`:
```python
# Queries always scope to tenant namespace
results = vector_store.query(
    query_vector=embedding,
    namespace=get_tenant_namespace(),  # from ContextVar
    top_k=10,
)
```

### Layer 3: API
JWT/API key validation extracts the tenant ID. The `TenantMiddleware` rejects requests without a valid tenant context before they reach any business logic.

## Why Namespace-Per-Tenant?
Alternatives considered:
- **Metadata filtering**: Single index, filter by `tenant_id` field → risk of filter bypass
- **Index-per-tenant**: Separate Pinecone index per tenant → expensive, hard to manage at 1000+ tenants
- **Namespace-per-tenant**: Logical isolation within a single index → Pinecone guarantees queries only search within one namespace

Namespace-per-tenant gives us the isolation guarantees of separate indexes without the operational overhead.

## Consequences
- Defense in depth: even if one layer fails, the others catch it
- Per-tenant config (confidence thresholds, model preferences) is straightforward
- Adding a new tenant is just creating a new namespace — no infrastructure changes
