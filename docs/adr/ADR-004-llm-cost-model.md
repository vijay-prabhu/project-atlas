# ADR-004: LLM Cost Model and Model Routing Strategy

## Status
Accepted

## Context
At scale (1B+ emails), LLM costs can dominate the operating budget. We need to optimize cost without sacrificing quality.

## Decision
Implement **task-based model routing** that picks the cheapest model meeting the quality bar for each task.

## Routing Rules

| Task | Model | Cost (per 1M input) | Latency (p50) | Why |
|------|-------|---------------------|---------------|-----|
| Email classification | gpt-4o-mini | $0.15 | ~200ms | Choosing from a known set of categories. High accuracy even with small model. |
| Metadata extraction | gpt-4o | $2.50 | ~800ms | Needs deeper understanding — parsing RFI refs, spec sections, ambiguous context. |
| Search synthesis | claude-sonnet-4 | $3.00 | ~1.2s | Good at structured reasoning with citations. |
| Simple Q&A | gpt-4o-mini | $0.15 | ~200ms | FAQ-style answers don't need heavy reasoning. |

## Cost Projections

For 10,000 emails/day:
- **Without routing**: All calls to gpt-4o → ~$75/day
- **With routing**: 70% to gpt-4o-mini, 30% to gpt-4o → ~$15/day
- **Savings**: 80% cost reduction

Additional optimizations:
- **Semantic cache**: Skip LLM for repeated patterns → 15-20% call reduction
- **Rule-based fast path**: Obvious classifications bypass LLM entirely → 30-40% call reduction
- **Context pruning**: Send top 5 chunks instead of 10 → 40% token reduction per RAG call

## How to Explain This to Non-Technical Stakeholders

"Think of it like having a team with a junior and senior analyst. Most emails are straightforward — the junior handles them quickly and cheaply. Only the tricky ones go to the senior. The result is the same quality but at a fraction of the cost."

Cost table for a PM/VP conversation:

| Monthly Volume | Without Optimization | With Optimization | Savings |
|---------------|---------------------|-------------------|---------|
| 100K emails | $2,250 | $450 | $1,800 |
| 500K emails | $11,250 | $2,250 | $9,000 |
| 1M emails | $22,500 | $4,500 | $18,000 |

## Consequences
- Model routing logic adds complexity — need to maintain routing rules
- Quality monitoring per model is essential — track accuracy by task type
- Fallback chain needed if primary model is unavailable
