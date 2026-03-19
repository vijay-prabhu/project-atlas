"""Search API routes."""

from fastapi import APIRouter, Depends

from src.agents.search_agent import run_search_agent
from src.api.deps import get_tenant_id
from src.api.schemas import SearchQuery, SearchResponse, SearchResult, Citation

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchQuery,
    tenant_id: str = Depends(get_tenant_id),
):
    """Run a semantic search across project documents.

    Supports three modes:
    - hybrid (default): BM25 keyword + vector semantic, merged with RRF
    - semantic: vector similarity only
    - keyword: BM25 keyword only
    """
    result = run_search_agent(
        query=request.query,
        tenant_id=tenant_id,
        top_k=request.top_k,
    )

    search_results = [
        SearchResult(
            chunk_text=chunk.get("text", ""),
            score=chunk.get("score", 0.0),
            source_type=chunk.get("source_type", "email"),
            source_id=chunk.get("id", ""),
            source_title=chunk.get("title", ""),
            metadata=chunk.get("metadata", {}),
        )
        for chunk in result.source_chunks
    ]

    citations = [
        Citation(
            claim=c.get("claim", ""),
            source_document=c.get("source_document", ""),
            source_chunk=c.get("source_chunk", ""),
            relevance_score=c.get("relevance_score", 0.0),
            verified=c.get("verified", False),
        )
        for c in result.citations
    ]

    return SearchResponse(
        answer=result.answer,
        results=search_results,
        citations=citations,
        search_type_used=result.search_type,
        total_found=len(search_results),
        confidence=result.confidence,
    )
