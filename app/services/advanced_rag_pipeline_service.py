"""
Advanced RAG Pipeline Service
User Query -> Rewrite/Expand -> BM25 (Sparse) -> Vector (Dense) -> Merge -> Cross-Encoder -> Top-K -> LLM
"""

from typing import List, Dict, Any
import logging
import asyncio
from openai import AsyncOpenAI
from app.config import settings
from app.services.query_rewriting_service import QueryRewritingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.bm25_service import BM25Service
from app.services.cross_encoder_reranking_service import CrossEncoderRerankingService

logger = logging.getLogger("rag_app.advanced_rag_pipeline_service")


class AdvancedRAGPipelineService:
    """End-to-end pipeline implementing the requested architecture."""

    def __init__(self, api_key: str | None = None, query_cache_service=None):
        """
        Initialize the pipeline service.

        Args:
            api_key: OpenAI API key (optional, uses settings if not provided)
            query_cache_service: Optional QueryCacheService for embedding caching
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")

        self.query_rewriting_service = QueryRewritingService(api_key=self.api_key)
        self.embedding_service = EmbeddingService(
            api_key=self.api_key, query_cache_service=query_cache_service
        )
        self.vector_service = VectorService()
        self.bm25_service = BM25Service()
        self.reranker = CrossEncoderRerankingService()
        self.llm_client = AsyncOpenAI(api_key=self.api_key)

        self.model = "gpt-4o-mini"
        self.temperature = 0.1
        self.max_tokens = 1000

    async def answer_query(
        self,
        query: str,
        top_k: int = 3,
        namespace: str = "default",
        filter_dict: Dict[str, Any] | None = None,
        use_bm25: bool = True,
        use_vector: bool = True,
        use_rerank: bool = True,
        use_llm: bool = True,
        use_rewrites: bool = True,
        use_expansions: bool = True,
        max_rewrites: int = 3,
        max_expansions: int = 5,
        sparse_top_k: int = 10,
        dense_top_k: int = 10,
        merge_top_k: int = 20,
        sparse_weight: float = 0.4,
        dense_weight: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Full pipeline with sparse + dense retrieval and cross-encoder reranking.

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

        # Step 1: Rewrite + Expand
        rewrites, expansions, expanded_query = [], [], query
        usage = {"rewrite": None, "expansion": None, "embedding": None}

        if use_rewrites and use_expansions:
            rewrite_data = await self.query_rewriting_service.rewrite_and_expand(
                query, max_rewrites=max_rewrites, max_expansions=max_expansions
            )
            rewrites = rewrite_data.get("rewrites", [])
            expansions = rewrite_data.get("expansions", [])
            expanded_query = rewrite_data.get("expanded_query", query)
            usage["rewrite"] = rewrite_data.get("usage", {}).get("rewrite")
            usage["expansion"] = rewrite_data.get("usage", {}).get("expansion")
        elif use_rewrites:
            rewrite_data = await self.query_rewriting_service.rewrite_query(
                query, max_rewrites=max_rewrites
            )
            rewrites = rewrite_data.get("rewrites", [])
            usage["rewrite"] = rewrite_data.get("usage")
        elif use_expansions:
            expansion_data = await self.query_rewriting_service.expand_query(
                query, max_expansions=max_expansions
            )
            expansions = expansion_data.get("expansions", [])
            expanded_query = expansion_data.get("expanded_query", query)
            usage["expansion"] = expansion_data.get("usage")

        query_variants = [query]
        query_variants.extend(rewrites)
        if use_expansions and expanded_query and expanded_query.strip() != query.strip():
            query_variants.append(expanded_query)
        query_variants = self._dedupe_variants(query_variants)

        # Step 2: Sparse retrieval (BM25) using expanded variants (async-safe)
        if use_bm25:
            sparse_results = await asyncio.to_thread(
                self._bm25_multi_search, query_variants, sparse_top_k
            )
        else:
            sparse_results = []

        # Step 3: Dense retrieval (Vector search) for all variants
        dense_results = []
        if use_vector:
            embeddings, embedding_usage = await self.embedding_service.generate_embeddings(
                query_variants
            )
            usage["embedding"] = embedding_usage

            for variant, embedding in zip(query_variants, embeddings):
                results = await self.vector_service.search(
                    query_embedding=embedding,
                    top_k=dense_top_k,
                    namespace=namespace,
                    filter_dict=filter_dict,
                )
                for chunk in results.get("chunks", []):
                    dense_results.append({**chunk, "matched_query": variant})
        else:
            usage["embedding"] = None

        # Step 4: Merge sparse + dense
        merged = self._merge_results(sparse_results, dense_results, sparse_weight, dense_weight)
        merged = sorted(merged.values(), key=lambda x: x.get("hybrid_score", 0.0), reverse=True)[
            :merge_top_k
        ]

        # Step 5: Cross-encoder re-ranking
        if use_rerank:
            reranked = (await self.reranker.rerank(query, merged, top_k=top_k)).get(
                "reranked_chunks", []
            )
        else:
            reranked = self._finalize_scores(merged)[:top_k]

        # Step 6: Build context
        context = self._build_context(reranked)

        if not context.strip():
            return {
                "question": query,
                "answer": "I don't have enough information to answer that question. Please upload relevant documents.",
                "sources": [],
                "chunks_used": 0,
                "model": self.model,
                "usage": usage,
            }

        llm_usage = None
        answer = None

        # Step 7: LLM answer (optional)
        if use_llm:
            prompt = self._create_prompt(query, context)
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. "
                        "If the context doesn't contain enough information, say so explicitly.",
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
            "sources": self._format_sources(reranked),
            "chunks_used": len(reranked),
            "model": self.model,
            "context": context if not use_llm else None,
            "usage": {
                "rewrite": usage.get("rewrite"),
                "expansion": usage.get("expansion"),
                "embedding": usage.get("embedding"),
                "llm": llm_usage,
            },
        }

    # ==================== Internal Helpers ====================

    def _dedupe_variants(self, variants: List[str]) -> List[str]:
        seen = set()
        result = []
        for v in variants:
            normalized = v.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            result.append(normalized)
        return result

    def _merge_results(
        self,
        sparse_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        sparse_weight: float,
        dense_weight: float,
    ) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}

        # Normalize scores (raw -> normalized once)
        max_sparse = max((c.get("sparse_score", 0.0) for c in sparse_results), default=1.0) or 1.0
        max_dense = max((c.get("score", 0.0) for c in dense_results), default=1.0) or 1.0

        for chunk in sparse_results:
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue
            sparse_score_raw = chunk.get("sparse_score", 0.0)
            sparse_score_norm = sparse_score_raw / max_sparse
            merged[chunk_id] = {
                **chunk,
                "sparse_score": sparse_score_raw,
                "dense_score": 0.0,
                "hybrid_score": sparse_weight * sparse_score_norm,
                "matched_queries": [],
            }

        for chunk in dense_results:
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue
            dense_score_raw = chunk.get("score", 0.0)
            dense_score_norm = dense_score_raw / max_dense

            if chunk_id not in merged:
                merged[chunk_id] = {
                    **chunk,
                    "sparse_score": 0.0,
                    "dense_score": dense_score_raw,
                    "hybrid_score": dense_weight * dense_score_norm,
                    "matched_queries": [chunk.get("matched_query")],
                }
            else:
                existing = merged[chunk_id]
                # Keep best dense score
                if dense_score_raw > existing.get("dense_score", 0.0):
                    existing.update(chunk)
                    existing["dense_score"] = dense_score_raw
                existing["hybrid_score"] = (
                    sparse_weight * (existing.get("sparse_score", 0.0) / max_sparse)
                    + dense_weight * dense_score_norm
                )
                mq = chunk.get("matched_query")
                if mq and mq not in existing.get("matched_queries", []):
                    existing["matched_queries"].append(mq)

        return merged

    def _bm25_multi_search(self, queries: List[str], top_k: int) -> List[Dict[str, Any]]:
        """
        Run BM25 for multiple query variants and merge by best sparse score.
        """
        merged: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            results = self.bm25_service.search(q, top_k=top_k)
            for chunk in results:
                chunk_id = chunk.get("id")
                if not chunk_id:
                    continue
                existing = merged.get(chunk_id)
                if not existing or chunk.get("sparse_score", 0.0) > existing.get(
                    "sparse_score", 0.0
                ):
                    merged[chunk_id] = chunk
        return list(merged.values())

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get("metadata", {}).get("filename", "Unknown")
            text = chunk.get("text", "")
            score = chunk.get("final_score", chunk.get("rerank_score", chunk.get("score", 0.0)))
            context_parts.append(f"[{i}] {filename} (score: {score:.4f})\n{text}")
        return "\n\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer based only on the context above."
        )

    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for chunk in chunks:
            sources.append(
                {
                    "id": chunk.get("id"),
                    "filename": chunk.get("metadata", {}).get("filename", ""),
                    "chunk_index": chunk.get("metadata", {}).get("chunk_index", 0),
                    "score": chunk.get(
                        "final_score", chunk.get("rerank_score", chunk.get("score", 0.0))
                    ),
                    "text": chunk.get("text", ""),
                }
            )
        return sources

    def _finalize_scores(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        finalized = []
        for chunk in chunks:
            score = chunk.get("hybrid_score", chunk.get("score", chunk.get("sparse_score", 0.0)))
            finalized.append(
                {
                    **chunk,
                    "vector_score": chunk.get(
                        "vector_score", chunk.get("score", chunk.get("dense_score", 0.0))
                    ),
                    "rerank_score": None,
                    "final_score": score,
                }
            )
        return finalized
