"""
Advanced RAG Pipeline Service
User Query -> Rewrite/Expand -> Hybrid Search Service (RRF) -> Top-K -> LLM
"""

from typing import List, Dict, Any, Protocol
import logging
import asyncio
import re
import numpy as np
from openai import AsyncOpenAI
from app.config import settings
from app.models.retrieved_chunks import RetrievedChunk
from app.services.query_rewriting_service import QueryRewritingService
from app.services.embedding_service import EmbeddingService
from app.services.hybdrid_retrieval_service import HybridSearchService, SearchResult


logger = logging.getLogger("rag_app.advanced_rag_pipeline_service")


class AdvancedRAGPipelineService:
    """
    End-to-end pipeline: Query -> Rewrite/Expand -> Hybrid Search -> LLM
    """

    def __init__(
        self,
        api_key: str | None = None,
        hybrid_search_service: HybridSearchService | None = None,
    ):
        """
        Initialize the pipeline service.

        Args:
            api_key: OpenAI API key (optional, uses settings if not provided)
            hybrid_search_service: Optional pre-configured HybridSearchService
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")

        self.query_rewriting_service = QueryRewritingService(api_key=self.api_key)
        self.embedding_service = EmbeddingService(api_key=self.api_key)
        self.llm_client = AsyncOpenAI(api_key=self.api_key)

        # Use provided hybrid search service or create default
        if hybrid_search_service:
            self.hybrid_search = hybrid_search_service
        else:
            self.hybrid_search = HybridSearchService(
                embedding_service=self.embedding_service,
                settings=settings,
                dense_backend="local",
                sparse_backend="bm25",
                fusion_strategy="rrf",
            )

        self.model = "gpt-4o-mini"
        self.temperature = 0.1
        self.max_tokens = 1000

    async def index_documents(self, chunks: list[RetrievedChunk]) -> None:
        """Index documents to both dense (Pinecone) and sparse (BM25) backends"""

        from langchain_core.documents import Document

        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.content,
                metadata={
                    "id": chunk.id,
                    **chunk.metadata,
                    "embedding": chunk.embedding,  # Pass embedding through metadata
                },
                id=chunk.id,
            )
            documents.append(doc)

        # Index to hybrid service (both Pinecone + BM25)
        self.hybrid_search.index(documents)

    async def answer_query(
        self,
        query: str,
        top_k: int = 5,
        use_rewrites: bool = True,
        use_expansions: bool = True,
        max_rewrites: int = 3,
        max_expansions: int = 5,
        sparse_weight: float | None = None,
        dense_weight: float | None = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline with hybrid search (RRF fusion, no reranking).

        Returns:
            Dictionary with answer, sources, and usage data.
        """
        if not query or not query.strip():
            return {
                "question": query,
                "answer": "Question cannot be empty.",
                "sources": [],
                "chunks_used": 0,
                "model": self.model,
                "usage": None,
            }

        query_variants = await self._generate_query_variants(
            query, use_rewrites, use_expansions, max_rewrites, max_expansions
        )

        variant_embeddings = await self._embed_query_variants(query_variants)

        all_results = []
        for variant, embedding in zip(query_variants, variant_embeddings):
            results = await self.hybrid_search.search(
                query=variant,
                query_embedding=embedding,
                top_k=top_k,
            )
            all_results.extend(results)

        seen: dict[str, SearchResult] = {}
        for result in all_results:
            cid = result.chunk.id
            if not cid:
                continue

            if cid not in seen or result.combined_score > seen[cid].combined_score:
                seen[cid] = result

        final_results = sorted(seen.values(), key=lambda x: x.combined_score, reverse=True)[:top_k]

        context = self._build_context(final_results)

        if not context.strip():
            return {
                "question": query,
                "answer": "I don't have enough information to answer that question. Please upload relevant documents.",
                "sources": [],
                "chunks_used": 0,
                "model": self.model,
                "retrieval_stats": {
                    "query_variants": len(query),
                    "unique_results": len(seen),
                    "final_results": len(final_results),
                },
                "usage": None,
            }

        # Step 5: LLM answer
        prompt = self._create_prompt(query, context)
        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context. "
                    "Use only the supplied context. If the exact answer is missing, provide the closest "
                    "supported partial answer and clearly state what is missing.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content
        llm_usage = (
            {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if hasattr(response, "usage") and response.usage
            else None
        )

        return {
            "question": query,
            "answer": answer,
            "sources": self._format_sources(final_results),
            "chunks_used": len(final_results),
            "model": self.model,
            "retrieval_stats": {
                "query_variants": len(query),
                "unique_results": len(seen),
                "final_results": len(final_results),
            },
            "usage": {
                "llm": llm_usage,
            },
        }

    # In advanced_rag_pipeline_service.py

    async def _embed_query_variants(self, variants: list[str]) -> list[list[float] | None]:
        """
        Batch embed all query variants for efficiency.
        """
        if not variants:
            return []

        try:
            # Call generate_embeddings directly with the list
            embeddings, _ = await self.embedding_service.generate_embeddings(variants)
            return embeddings
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}, falling back to individual")
            embeddings = []
            for variant in variants:
                try:
                    result, _ = await self.embedding_service.generate_embeddings([variant])
                    embeddings.append(result[0] if result else None)
                except Exception:
                    embeddings.append(None)
            return embeddings

    async def _generate_query_variants(
        self,
        query: str,
        use_rewrites: bool,
        use_expansions: bool,
        max_rewrites: int,
        max_expansions: int,
    ) -> list[str]:
        """Generate query variants through rewriting and expansion"""
        variants = [query]

        if use_rewrites and use_expansions:
            rewrite_data = await self.query_rewriting_service.rewrite_and_expand(
                query, max_rewrites=max_rewrites, max_expansions=max_expansions
            )
            variants.extend(rewrite_data.get("rewrites", []))
            variants.extend(rewrite_data.get("expansions", []))
            # DON'T add expanded_query - it's redundant with individual expansions
            # The expanded_query is just query + " " + " ".join(expansions)

        elif use_rewrites:
            rewrite_data = await self.query_rewriting_service.rewrite_query(
                query, max_rewrites=max_rewrites
            )
            variants.extend(rewrite_data.get("rewrites", []))

        elif use_expansions:
            expansion_data = await self.query_rewriting_service.expand_query(
                query, max_expansions=max_expansions
            )
            variants.extend(expansion_data.get("expansions", []))
            # Only add expanded_query when NOT using individual expansions
            expanded = expansion_data.get("expanded_query")
            if expanded and expanded != query:
                variants.append(expanded)

        # Deduplicate (keep order, case-insensitive)
        seen = set()
        result = []
        for v in variants:
            normalized = v.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(v.strip())

        return result

    def _build_context(self, search_results: list[SearchResult]) -> str:
        """Build context string from SearchResult objects"""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            chunk = result.chunk

            metadata = chunk.metadata or {}
            filename = metadata.get("filename", "Unknown")

            text = getattr(chunk, "content", None) or getattr(chunk, "text", "")

            score = result.combined_score
            source = result.source

            context_parts.append(f"[{i}] {filename} (score: {score:.4f}, source: {source})\n{text}")

        return "\n\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """Create LLM prompt"""
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer based only on the context above. If the exact detail is not present, provide the "
            "closest supported partial answer and clearly state what is missing."
        )

    def _format_sources(self, search_results: list[SearchResult]) -> list[dict]:
        """Format SearchResult objects as sources"""
        sources = []
        for result in search_results:
            chunk = result.chunk
            metadata = chunk.metadata or {}

            # Get text content safely
            text = getattr(chunk, "content", "") or getattr(chunk, "text", "")

            sources.append(
                {
                    "id": chunk.id,
                    "filename": metadata.get("filename", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "score": result.combined_score,  # Use combined_score from SearchResult
                    "source_type": result.source,  # Use source from SearchResult
                    "dense_score": result.dense_score,
                    "sparse_score": result.sparse_score,
                    "text": text[:200] + "..." if len(text) > 200 else text,
                }
            )

        return sources
