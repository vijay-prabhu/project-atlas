"""Core RAG pipeline: retrieve → rerank → generate → cite → verify.

This is the main entry point for answering questions. It pulls chunks from
the vector store, reranks them, builds a prompt with source references,
generates an answer with citations, then verifies each citation against the
actual source text.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.core.observability import AgentTrace, get_logger, trace_step
from src.pipeline.embedding import EmbeddingService
from src.rag.citations import Citation, extract_citations, verify_all_claims

logger = get_logger(__name__)

# System prompt that tells the LLM to cite its sources
_SYSTEM_PROMPT = """You are an AECO (Architecture, Engineering, Construction, Owner) domain expert assistant.

Answer the user's question based ONLY on the provided context. Do not use prior knowledge.

Rules:
1. Cite every factual claim using [Source: document_name, section] format
2. If the context does not contain enough information, say so explicitly
3. Be precise — use exact numbers, dates, and spec references from the sources
4. Keep answers concise and actionable for construction professionals

Context from retrieved documents:
{context}"""


@dataclass
class SourceChunk:
    """A retrieved chunk with its metadata and relevance score."""

    text: str
    score: float
    source_document: str = ""
    source_section: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    source_chunks: list[SourceChunk] = field(default_factory=list)
    confidence: float = 0.0
    trace: dict = field(default_factory=dict)


class RAGPipeline:
    """Orchestrates retrieval, reranking, generation, and citation verification.

    Dependencies are injected so you can swap out the vector store, LLM client,
    and reranker without changing the pipeline logic.
    """

    def __init__(
        self,
        vector_store: Any,
        llm_client: Any,
        reranker: Any = None,
        embedding_service: Optional[EmbeddingService] = None,
        top_k: int = 10,
        top_n_rerank: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._llm_client = llm_client
        self._reranker = reranker
        self._embedding_service = embedding_service
        self._top_k = top_k
        self._top_n_rerank = top_n_rerank

    def query(
        self,
        question: str,
        tenant_id: str,
        filters: Optional[dict] = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline: embed → retrieve → rerank → generate → cite → verify.

        Args:
            question: The user's question in natural language.
            tenant_id: Tenant namespace for vector store isolation.
            filters: Optional metadata filters (e.g. {"doc_type": "email"}).

        Returns:
            RAGResponse with the answer, verified citations, source chunks, and trace.
        """
        trace = AgentTrace()
        start = time.perf_counter()

        # 1. Embed the question
        with trace_step(trace, "rag", "embed_question") as meta:
            embedding_svc = self._embedding_service or EmbeddingService(
                tenant_id=tenant_id,
            )
            question_embedding = embedding_svc.embed_text(question)
            meta["embedding_dim"] = len(question_embedding)

        # 2. Retrieve top-K chunks from vector store
        with trace_step(trace, "rag", "retrieve") as meta:
            raw_results = self._retrieve(
                question_embedding, tenant_id, filters,
            )
            meta["retrieved_count"] = len(raw_results)

        # 3. Rerank results
        with trace_step(trace, "rag", "rerank") as meta:
            ranked_chunks = self._rerank(question, raw_results)
            meta["reranked_count"] = len(ranked_chunks)

        # 4. Build context from top chunks with source references
        context = self._build_context(ranked_chunks)

        # 5. Generate answer via LLM
        with trace_step(trace, "rag", "generate") as meta:
            answer = self._generate(question, context)
            meta["answer_length"] = len(answer)

        # 6. Extract citations from the answer
        with trace_step(trace, "rag", "extract_citations") as meta:
            citations = extract_citations(answer, ranked_chunks)
            meta["citation_count"] = len(citations)

        # 7. Verify each citation against source chunks
        with trace_step(trace, "rag", "verify_citations") as meta:
            verified_citations = verify_all_claims(citations)
            verified_count = sum(1 for c in verified_citations if c.verified)
            meta["verified_count"] = verified_count
            meta["total_citations"] = len(verified_citations)

        # Calculate confidence based on citation verification rate
        confidence = 0.0
        if verified_citations:
            confidence = verified_count / len(verified_citations)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "RAG query completed",
            extra={
                "tenant_id": tenant_id,
                "question_length": len(question),
                "chunks_retrieved": len(raw_results),
                "citations": len(verified_citations),
                "confidence": round(confidence, 3),
                "duration_ms": duration_ms,
            },
        )

        return RAGResponse(
            answer=answer,
            citations=verified_citations,
            source_chunks=ranked_chunks,
            confidence=round(confidence, 3),
            trace=trace.to_dict(),
        )

    def _retrieve(
        self,
        query_embedding: list[float],
        tenant_id: str,
        filters: Optional[dict] = None,
    ) -> list[SourceChunk]:
        """Query the vector store and convert results to SourceChunk objects."""
        query_params: dict[str, Any] = {
            "vector": query_embedding,
            "top_k": self._top_k,
            "namespace": tenant_id,
            "include_metadata": True,
        }
        if filters:
            query_params["filter"] = filters

        response = self._vector_store.query(**query_params)
        matches = response.get("matches", [])

        chunks: list[SourceChunk] = []
        for match in matches:
            metadata = match.get("metadata", {})
            chunks.append(SourceChunk(
                text=metadata.get("text", ""),
                score=match.get("score", 0.0),
                source_document=metadata.get("source_document", metadata.get("subject", "")),
                source_section=metadata.get("section", metadata.get("section_header", "")),
                metadata=metadata,
            ))

        return chunks

    def _rerank(
        self,
        question: str,
        chunks: list[SourceChunk],
    ) -> list[SourceChunk]:
        """Rerank chunks using the reranker, or fall back to score-based ordering."""
        if self._reranker is not None:
            texts = [chunk.text for chunk in chunks]
            rerank_results = self._reranker.rerank(
                query=question,
                documents=texts,
                top_n=self._top_n_rerank,
            )

            reranked: list[SourceChunk] = []
            for result in rerank_results:
                idx = result.get("index", 0)
                if idx < len(chunks):
                    chunk = chunks[idx]
                    chunk.score = result.get("relevance_score", chunk.score)
                    reranked.append(chunk)

            return reranked

        # No reranker — just sort by score and take top N
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        return sorted_chunks[: self._top_n_rerank]

    def _build_context(self, chunks: list[SourceChunk]) -> str:
        """Build a context string from chunks with source labels.

        Each chunk is labeled with its source document and section so the
        LLM can reference them in citations.
        """
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            source_label = chunk.source_document or f"Document {i}"
            section_label = f", {chunk.source_section}" if chunk.source_section else ""

            parts.append(
                f"[Source {i}: {source_label}{section_label}]\n{chunk.text}"
            )

        return "\n\n---\n\n".join(parts)

    def _generate(self, question: str, context: str) -> str:
        """Generate an answer using the LLM client."""
        system_prompt = _SYSTEM_PROMPT.format(context=context)

        response = self._llm_client.generate(
            prompt=question,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=2048,
        )

        return response.content
