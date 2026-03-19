# ADR-002: AECO-Specific Chunking Strategy

## Status
Accepted

## Context
AECO documents come in multiple formats with different structural patterns. A single chunking strategy won't work well across emails, specifications, and drawing notes.

## Decision
Implement three chunking strategies, each optimized for its document type:

### 1. Email Chunking (chunk_email)
- **Chunk size**: 512 tokens, 50 token overlap
- **Separators**: Reply chain boundaries ("From:", "On ... wrote:"), signature blocks
- **Rationale**: Emails are conversational. Preserving reply context is critical — a reply saying "approved" only makes sense with the original question.

### 2. Specification Chunking (chunk_specification)
- **Chunk size**: 1024 tokens, 100 token overlap
- **Separators**: CSI section headers ("PART 1 - GENERAL", "2.01 MATERIALS")
- **Rationale**: Specs are dense, hierarchical documents. Larger chunks preserve section context. Never split mid-section — a spec clause needs its full context.

### 3. Drawing Notes Chunking (chunk_drawing_notes)
- **Chunk size**: 256 tokens, 25 token overlap
- **Separators**: Sheet references, callout numbers, note markers
- **Rationale**: Drawing notes are short, discrete annotations. Smaller chunks keep each note self-contained.

## Why Not One Strategy?
We tested a single 512-token strategy across all document types. Results:
- Emails: Good — natural paragraph boundaries worked
- Specs: Poor — sections were split mid-clause, losing critical context
- Drawing notes: Poor — multiple unrelated notes merged into single chunks

## Consequences
- Three code paths to maintain
- Need to detect document type before chunking (handled by `chunk_auto()`)
- Each strategy needs its own eval set to verify quality
