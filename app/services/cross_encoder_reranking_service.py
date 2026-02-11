"""
Cross Encoder Re-ranking Service
Pipeline: User Query -> Query Rewriting / Expansion -> Hybrid Retrieval -> Cross Encoder Re-ranking
"""

from typing import List, Dict, Any, Tuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("rag_app.cross_encoder_reranking_service")


class CrossEncoderRerankingService:
    """Service that re-ranks retrieved chunks using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_candidates: int = 20,
        executor: ThreadPoolExecutor | None = None,
    ):
        """
        Initialize the cross-encoder re-ranking service.

        Args:
            model_name: Sentence-Transformers cross-encoder model name
            batch_size: Batch size for cross-encoder prediction
            max_candidates: Max number of candidates to rerank
            executor: Optional thread pool for non-blocking inference
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        self.model = None
        self.executor = executor

        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            logger.info(f"Cross-encoder model loaded: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Cross-encoder re-ranking is disabled. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder model '{self.model_name}': {e}")

    def is_available(self) -> bool:
        """Check if the cross-encoder model is available."""
        return self.model is not None

    async def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 3,
        max_candidates: int | None = None,
        batch_size: int | None = None,
    ) -> Dict[str, Any]:
        """
        Re-rank chunks with a cross-encoder (async-safe).

        Args:
            query: Original user query
            chunks: Retrieved chunks (from hybrid retrieval)
            top_k: Number of top-ranked results to return
            max_candidates: Max number of items to rerank (pre-filter)
            batch_size: Batch size for predict()

        Returns:
            Dictionary containing:
                - query: Original query
                - reranked_chunks: Chunks ordered by final_score
                - total_reranked: Number of results returned
                - model: Cross-encoder model used (or None)
        """
        if not query or not query.strip() or not chunks:
            return {
                "query": query,
                "reranked_chunks": [],
                "total_reranked": 0,
                "model": self.model_name if self.model else None,
            }

        if not self.model:
            # No cross-encoder available; return original ordering
            return {
                "query": query,
                "reranked_chunks": self._preserve_scores(chunks[:top_k]),
                "total_reranked": min(len(chunks), top_k),
                "model": None,
            }

        try:
            candidates = self._select_candidates(chunks, max_candidates or self.max_candidates)
            batch = batch_size or self.batch_size

            # Run blocking model in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                self.executor, self._predict_scores, query, candidates, batch
            )

            scored_chunks: List[Tuple[float, Dict[str, Any]]] = []
            for score, chunk in zip(scores, candidates):
                vector_score = self._extract_vector_score(chunk)
                enriched = {
                    **chunk,
                    "vector_score": vector_score,
                    "rerank_score": float(score),
                    "final_score": float(score),
                }
                scored_chunks.append((float(score), enriched))

            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            reranked = [chunk for _, chunk in scored_chunks[:top_k]]

            return {
                "query": query,
                "reranked_chunks": reranked,
                "total_reranked": len(reranked),
                "model": self.model_name,
            }
        except Exception as e:
            raise Exception(f"Cross-encoder re-ranking failed: {str(e)}")

    # ==================== Internal Helpers ====================

    def _predict_scores(self, query: str, chunks: List[Dict[str, Any]], batch_size: int):
        pairs = [(query, chunk.get("text", "")) for chunk in chunks]
        return self.model.predict(pairs, batch_size=batch_size)

    def _select_candidates(
        self, chunks: List[Dict[str, Any]], max_candidates: int
    ) -> List[Dict[str, Any]]:
        if max_candidates <= 0:
            return []
        # Pre-filter by existing dense/hybrid score to limit reranking cost
        ranked = sorted(
            chunks, key=lambda x: x.get("hybrid_score", x.get("score", 0.0)), reverse=True
        )
        return ranked[:max_candidates]

    def _extract_vector_score(self, chunk: Dict[str, Any]) -> float:
        if "score" in chunk:
            return float(chunk.get("score", 0.0))
        if "dense_score" in chunk:
            return float(chunk.get("dense_score", 0.0))
        if "hybrid_score" in chunk:
            return float(chunk.get("hybrid_score", 0.0))
        return 0.0

    def _preserve_scores(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        preserved = []
        for chunk in chunks:
            vector_score = self._extract_vector_score(chunk)
            preserved.append(
                {
                    **chunk,
                    "vector_score": vector_score,
                    "rerank_score": None,
                    "final_score": vector_score,
                }
            )
        return preserved
