"""
Hybrid Retrieval Service
Pipeline: User Query -> Query Rewriting / Expansion -> Hybrid Retrieval
"""

from typing import List, Dict, Any
import logging
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.query_rewriting_service import QueryRewritingService
from app.config import settings

logger = logging.getLogger("rag_app.hybrid_retrieval_service")


class HybridRetrievalService:
    """Service that combines query rewriting/expansion with vector retrieval."""

    def __init__(self, api_key: str | None = None, query_cache_service=None):
        """
        Initialize the hybrid retrieval service.

        Args:
            api_key: OpenAI API key (optional, uses settings if not provided)
            query_cache_service: Optional QueryCacheService for embedding caching
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")

        self.embedding_service = EmbeddingService(
            api_key=self.api_key,
            query_cache_service=query_cache_service
        )
        self.vector_service = VectorService()
        self.query_rewriting_service = QueryRewritingService(api_key=self.api_key)

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
        namespace: str = "default",
        filter_dict: Dict[str, Any] | None = None,
        use_rewrites: bool = True,
        use_expansions: bool = True,
        max_rewrites: int = 3,
        max_expansions: int = 5
    ) -> Dict[str, Any]:
        """
        Run hybrid retrieval with query rewriting and expansion.

        Args:
            query: Original user query
            top_k: Number of final results to return
            namespace: Pinecone namespace to search
            filter_dict: Optional metadata filter for vector search
            use_rewrites: Whether to generate rewrites
            use_expansions: Whether to generate expansions
            max_rewrites: Max number of rewrites
            max_expansions: Max number of expansions

        Returns:
            Dictionary containing:
                - original_query
                - query_variants
                - rewrites
                - expansions
                - chunks
                - total_found
                - usage
        """
        if not query or not query.strip():
            return {
                "original_query": query,
                "query_variants": [],
                "rewrites": [],
                "expansions": [],
                "chunks": [],
                "total_found": 0,
                "usage": None,
            }

        # Build query variants
        rewrites, expansions, expanded_query = [], [], query
        usage = {"rewrite": None, "expansion": None, "embedding": None}

        try:
            if use_rewrites and use_expansions:
                rewrite_data = await self.query_rewriting_service.rewrite_and_expand(
                    query,
                    max_rewrites=max_rewrites,
                    max_expansions=max_expansions
                )
                rewrites = rewrite_data.get("rewrites", [])
                expansions = rewrite_data.get("expansions", [])
                expanded_query = rewrite_data.get("expanded_query", query)
                usage["rewrite"] = rewrite_data.get("usage", {}).get("rewrite")
                usage["expansion"] = rewrite_data.get("usage", {}).get("expansion")
            elif use_rewrites:
                rewrite_data = await self.query_rewriting_service.rewrite_query(
                    query,
                    max_rewrites=max_rewrites
                )
                rewrites = rewrite_data.get("rewrites", [])
                usage["rewrite"] = rewrite_data.get("usage")
            elif use_expansions:
                expansion_data = await self.query_rewriting_service.expand_query(
                    query,
                    max_expansions=max_expansions
                )
                expansions = expansion_data.get("expansions", [])
                expanded_query = expansion_data.get("expanded_query", query)
                usage["expansion"] = expansion_data.get("usage")

            query_variants = [query]
            query_variants.extend(rewrites)
            if use_expansions and expanded_query and expanded_query.strip() != query.strip():
                query_variants.append(expanded_query)
            query_variants = self._dedupe_variants(query_variants)

            # Generate embeddings for all query variants
            embeddings, embedding_usage = await self.embedding_service.generate_embeddings(query_variants)
            usage["embedding"] = embedding_usage

            # Search per variant and merge results
            merged_results = {}
            for variant, embedding in zip(query_variants, embeddings):
                search_results = await self.vector_service.search(
                    query_embedding=embedding,
                    top_k=top_k,
                    namespace=namespace,
                    filter_dict=filter_dict
                )
                self._merge_results(merged_results, search_results.get("chunks", []), variant)

            # Sort by best score across variants
            ranked = sorted(
                merged_results.values(),
                key=lambda x: x.get("score", 0.0),
                reverse=True
            )[:top_k]

            return {
                "original_query": query,
                "query_variants": query_variants,
                "rewrites": rewrites,
                "expansions": expansions,
                "chunks": ranked,
                "total_found": len(ranked),
                "usage": usage,
            }
        except Exception as e:
            raise Exception(f"Hybrid retrieval failed: {str(e)}")

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
        merged_results: Dict[str, Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        variant: str
    ):
        for chunk in chunks:
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue

            existing = merged_results.get(chunk_id)
            if not existing:
                merged_results[chunk_id] = {
                    **chunk,
                    "matched_queries": [variant]
                }
            else:
                # Keep the highest score and track contributing queries
                if chunk.get("score", 0.0) > existing.get("score", 0.0):
                    existing.update(chunk)
                if variant not in existing.get("matched_queries", []):
                    existing["matched_queries"].append(variant)
