from typing import Protocol, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import Stemmer
from loguru import logger

from app.models.retrieved_chunks import RetrievedChunk
from langchain_core.documents import Document
from app.config import settings


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class SearchResult:
    """Unified search result from any backend"""

    chunk: RetrievedChunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    source: str = ""  # 'dense', 'sparse', or 'hybrid'


# ============================================================================
# PROTOCOLS & ABSTRACT BASE
# ============================================================================


class DenseBackend(Protocol):
    """Protocol for dense vector retrieval backends"""

    def search(
        self, query: str, query_embedding: list[float] | None, top_k: int
    ) -> list[RetrievedChunk]:
        """Retrieve top_k chunks using dense vectors"""
        ...

    def encode_query(self, query: str) -> list[float]:
        """Encode query to embedding vector"""
        ...


class SparseBackend(Protocol):
    """Protocol for sparse/BM25 retrieval backends"""

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve top_k chunks using sparse vectors/BM25"""
        ...

    def index_documents(self, documents: list[Document]) -> None:
        """Build or update the sparse index"""
        ...


class FusionStrategy(Protocol):
    """Protocol for merging dense and sparse results"""

    def fuse(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        dense_weight: float,
        sparse_weight: float,
        top_k: int,
    ) -> list[SearchResult]:
        """Merge two ranked lists into unified result"""
        ...


# ============================================================================
# DENSE BACKEND IMPLEMENTATIONS
# ============================================================================


class LocalDenseBackend:
    """Local sentence-transformers dense retrieval"""

    def __init__(self, settings):
        self.settings = settings
        self._model: SentenceTransformer | None = None
        self._chunk_index: list[RetrievedChunk] = []
        self._embeddings: np.ndarray | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._model is None:
            try:
                logger.info(f"Loading dense encoder: {self.settings.embedding_model}")
                self._model = SentenceTransformer(self.settings.embedding_model)
                logger.info("Dense encoder loaded")
            except Exception as e:
                logger.error(f"Failed to load dense encoder: {e}")
                raise
        return self._model

    def index_documents(self, chunks: list[RetrievedChunk]) -> None:
        """Pre-compute and store embeddings for all chunks"""
        logger.info(f"Indexing {len(chunks)} chunks for dense retrieval")
        self._chunk_index = chunks

        texts = [chunk.content for chunk in chunks]
        self._embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        logger.info("Dense indexing complete")

    def encode_query(self, query: str) -> list[float]:
        """Encode query to embedding"""
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    def search(
        self, query: str, query_embedding: list[float] | None, top_k: int
    ) -> list[RetrievedChunk]:
        """Cosine similarity search over pre-computed embeddings"""
        if self._embeddings is None:
            raise RuntimeError("Dense backend not indexed. Call index_documents() first.")

        # Encode query if embedding not provided
        if query_embedding is None:
            query_embedding = self.encode_query(query)

        query_vec = np.array(query_embedding)

        # Normalize for cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        embeddings_norm = self._embeddings / (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Compute similarities
        similarities = np.dot(embeddings_norm, query_norm)

        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            chunk = self._chunk_index[int(idx)]
            chunk.score = float(similarities[idx])
            results.append(chunk)

        logger.info(f"Dense retrieval: {len(results)} chunks")
        return results


class OpenAIEmbeddingBackend:
    """OpenAI API-based dense retrieval"""

    def __init__(self, settings):
        self.settings = settings
        self._chunk_index: list[RetrievedChunk] = []
        self._embeddings: list[list[float]] = []

        if not settings.openai_api_key:
            raise ValueError("OpenAI API key required")

        try:
            import openai

            self.client = openai.OpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI embedding backend initialized")
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")

    def index_documents(self, chunks: list[RetrievedChunk]) -> None:
        """Batch embed and store documents via OpenAI API"""
        logger.info(f"Indexing {len(chunks)} chunks with OpenAI")

        texts = [chunk.content for chunk in chunks]
        self._chunk_index = chunks

        # Batch processing with rate limiting
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.settings.openai_embedding_model or "text-embedding-3-small", input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        self._embeddings = all_embeddings
        logger.info("OpenAI indexing complete")

    def encode_query(self, query: str) -> list[float]:
        """Encode query via OpenAI API"""
        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model or "text-embedding-3-small", input=[query]
        )
        return response.data[0].embedding

    def search(
        self, query: str, query_embedding: list[float] | None, top_k: int
    ) -> list[RetrievedChunk]:
        """Cosine similarity over OpenAI embeddings"""
        if not self._embeddings:
            raise RuntimeError("Backend not indexed")

        if query_embedding is None:
            query_embedding = self.encode_query(query)

        query_vec = np.array(query_embedding)

        # Compute cosine similarities
        similarities = []
        for emb in self._embeddings:
            emb_vec = np.array(emb)
            sim = np.dot(query_vec, emb_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(emb_vec) + 1e-8
            )
            similarities.append(sim)

        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self._chunk_index[int(idx)]
            chunk.score = float(similarities[idx])
            results.append(chunk)

        return results


# ============================================================================
# SPARSE BACKEND IMPLEMENTATIONS
# ============================================================================


class BM25Backend:
    """Local BM25 sparse retrieval with stemming"""

    def __init__(self, settings):
        self.settings = settings
        self._chunk_index: list[RetrievedChunk] = []
        self._bm25: BM25Okapi | None = None
        self._stemmer = Stemmer.Stemmer("english")

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize and stem text for BM25"""
        # Simple whitespace tokenization + stemming
        tokens = text.lower().split()
        return self._stemmer.stemWords(tokens)

    def index_documents(self, documents: list[Document] | list[RetrievedChunk]) -> None:
        """Build BM25 index from documents"""
        logger.info(f"Building BM25 index for {len(documents)} documents")

        # Handle both Document and RetrievedChunk types
        if documents and hasattr(documents[0], "chunks"):
            # It's a list of Documents with sub-chunks
            all_chunks = []
            for doc in documents:
                all_chunks.extend(doc.chunks)
            self._chunk_index = all_chunks
        else:
            # It's already chunks
            self._chunk_index = documents

        # Tokenize all documents
        tokenized_docs = [self._tokenize(chunk.content) for chunk in self._chunk_index]

        self._bm25 = BM25Okapi(tokenized_docs)
        logger.info("BM25 index built")

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """BM25 retrieval"""
        if self._bm25 is None:
            raise RuntimeError("BM25 not indexed. Call index_documents() first.")

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return positive scores
                chunk = self._chunk_index[int(idx)]
                chunk.score = float(scores[idx])
                results.append(chunk)

        logger.info(f"BM25 retrieval: {len(results)} chunks")
        return results


class ElasticsearchSparseBackend:
    """Elasticsearch/Opensearch BM25 backend"""

    def __init__(self, settings):
        self.settings = settings
        self.host = settings.es_host or "localhost:9200"

        try:
            from elasticsearch import Elasticsearch

            self.client = Elasticsearch([self.host])
            self.index_name = settings.es_index or "documents"
            logger.info(f"ES backend initialized: {self.host}")
        except ImportError:
            raise ImportError("elasticsearch package required")

    def index_documents(self, documents: list[Document]) -> None:
        """Index documents to Elasticsearch"""
        # Bulk index implementation...
        pass  # Implementation depends on ES mapping

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """ES BM25 search"""
        es_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content^3", "title^2", "metadata.*"],
                    "type": "best_fields",
                }
            },
            "size": top_k,
        }

        response = self.client.search(index=self.index_name, body=es_query)

        results = []
        for hit in response["hits"]["hits"]:
            chunk = RetrievedChunk(
                id=hit["_id"],
                content=hit["_source"]["content"],
                metadata=hit["_source"].get("metadata", {}),
                score=hit["_score"],
            )
            results.append(chunk)

        return results


# ============================================================================
# FUSION STRATEGIES
# ============================================================================


class ReciprocalRankFusion:
    """
    RRF: Combines rankings using reciprocal rank formula.
    Score = Σ 1/(k + rank) for each list containing the item.
    k=60 is standard (tuneable).
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Merge using Reciprocal Rank Fusion"""

        scores: dict[str, dict] = {}  # chunk_id -> {dense_rank, sparse_rank, chunk}

        # Process dense results (rank 1-indexed)
        for rank, chunk in enumerate(dense_results, 1):
            cid = chunk.id
            if cid not in scores:
                scores[cid] = {"chunk": chunk, "dense_rank": None, "sparse_rank": None}
            scores[cid]["dense_rank"] = rank

        # Process sparse results
        for rank, chunk in enumerate(sparse_results, 1):
            cid = chunk.id
            if cid not in scores:
                scores[cid] = {"chunk": chunk, "dense_rank": None, "sparse_rank": None}
            scores[cid]["sparse_rank"] = rank

        # Calculate RRF scores
        fused_results = []
        for cid, data in scores.items():
            rrf_score = 0.0

            if data["dense_rank"]:
                rrf_score += dense_weight * (1.0 / (self.k + data["dense_rank"]))
            if data["sparse_rank"]:
                rrf_score += sparse_weight * (1.0 / (self.k + data["sparse_rank"]))

            # Determine source
            sources = []
            if data["dense_rank"]:
                sources.append("dense")
            if data["sparse_rank"]:
                sources.append("sparse")

            result = SearchResult(
                chunk=data["chunk"],
                dense_score=data["chunk"].score if data["dense_rank"] else 0.0,
                sparse_score=data["chunk"].score if data["sparse_rank"] else 0.0,
                combined_score=rrf_score,
                source="+".join(sources),
            )
            fused_results.append(result)

        # Sort by combined score and take top_k
        fused_results.sort(key=lambda x: x.combined_score, reverse=True)

        logger.info(
            f"RRF fusion: dense={len(dense_results)}, sparse={len(sparse_results)}, "
            f"unique={len(fused_results)}, final={min(top_k, len(fused_results))}"
        )

        return fused_results[:top_k]


class DistributionBasedScoreFusion:
    """
    DBSF: Normalizes scores to same distribution then weighted sum.
    Better when raw scores are comparable.
    """

    def fuse(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Merge using min-max normalization and weighted sum"""

        # Normalize dense scores to [0, 1]
        dense_scores = np.array([c.score for c in dense_results])
        if len(dense_scores) > 1:
            dense_norm = (dense_scores - dense_scores.min()) / (
                dense_scores.max() - dense_scores.min() + 1e-8
            )
        else:
            dense_norm = np.ones_like(dense_scores)

        dense_map = {chunk.id: score for chunk, score in zip(dense_results, dense_norm)}

        # Normalize sparse scores
        sparse_scores = np.array([c.score for c in sparse_results])
        if len(sparse_scores) > 1:
            sparse_norm = (sparse_scores - sparse_scores.min()) / (
                sparse_scores.max() - sparse_scores.min() + 1e-8
            )
        else:
            sparse_norm = np.ones_like(sparse_scores)

        sparse_map = {chunk.id: score for chunk, score in zip(sparse_results, sparse_norm)}

        # Combine all unique chunks
        all_ids = set(dense_map.keys()) | set(sparse_map.keys())

        fused_results = []
        for cid in all_ids:
            # Find original chunk
            chunk = next((c for c in dense_results + sparse_results if c.id == cid), None)

            d_score = dense_map.get(cid, 0.0)
            s_score = sparse_map.get(cid, 0.0)
            combined = dense_weight * d_score + sparse_weight * s_score

            sources = []
            if cid in dense_map:
                sources.append("dense")
            if cid in sparse_map:
                sources.append("sparse")

            fused_results.append(
                SearchResult(
                    chunk=chunk,
                    dense_score=float(d_score),
                    sparse_score=float(s_score),
                    combined_score=combined,
                    source="+".join(sources),
                )
            )

        fused_results.sort(key=lambda x: x.combined_score, reverse=True)
        return fused_results[:top_k]


class WeightedSumFusion:
    """Simple weighted sum of raw scores (when scales are comparable)"""

    def fuse(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Direct weighted sum without normalization"""

        score_map: dict[str, dict] = {}

        for chunk in dense_results:
            score_map.setdefault(chunk.id, {"chunk": chunk, "dense": 0.0, "sparse": 0.0})
            score_map[chunk.id]["dense"] = chunk.score

        for chunk in sparse_results:
            score_map.setdefault(chunk.id, {"chunk": chunk, "dense": 0.0, "sparse": 0.0})
            score_map[chunk.id]["sparse"] = chunk.score

        results = []
        for cid, data in score_map.items():
            combined = dense_weight * data["dense"] + sparse_weight * data["sparse"]

            sources = []
            if data["dense"] > 0:
                sources.append("dense")
            if data["sparse"] > 0:
                sources.append("sparse")

            results.append(
                SearchResult(
                    chunk=data["chunk"],
                    dense_score=data["dense"],
                    sparse_score=data["sparse"],
                    combined_score=combined,
                    source="+".join(sources),
                )
            )

        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]


# ============================================================================
# MAIN HYBRID SERVICE
# ============================================================================


class HybridSearchService:
    """
    Pluggable hybrid search combining dense + sparse retrieval with configurable fusion.

    Backends:
    - Dense: 'local' (sentence-transformers), 'openai' (API)
    - Sparse: 'bm25' (local), 'elasticsearch' (hosted)
    - Fusion: 'rrf' (Reciprocal Rank Fusion), 'dbsf' (Distribution-based), 'weighted'

    Config-driven via settings.hybrid_dense_backend, settings.hybrid_sparse_backend, etc.
    """

    DENSE_BACKENDS = {
        "local": LocalDenseBackend,
        "openai": OpenAIEmbeddingBackend,
    }

    SPARSE_BACKENDS = {
        "bm25": BM25Backend,
        "elasticsearch": ElasticsearchSparseBackend,
    }

    FUSION_STRATEGIES = {
        "rrf": ReciprocalRankFusion,
        "dbsf": DistributionBasedScoreFusion,
        "weighted": WeightedSumFusion,
    }

    def __init__(self):
        self.settings = settings

        # Initialize dense backend
        dense_backend_name = getattr(self.settings, "hybrid_dense_backend", "local")
        if dense_backend_name not in self.DENSE_BACKENDS:
            raise ValueError(f"Unknown dense backend: {dense_backend_name}")

        self.dense_backend = self.DENSE_BACKENDS[dense_backend_name](self.settings)
        logger.info(f"Dense backend: {dense_backend_name}")

        # Initialize sparse backend
        sparse_backend_name = getattr(self.settings, "hybrid_sparse_backend", "bm25")
        if sparse_backend_name not in self.SPARSE_BACKENDS:
            raise ValueError(f"Unknown sparse backend: {sparse_backend_name}")

        self.sparse_backend = self.SPARSE_BACKENDS[sparse_backend_name](self.settings)
        logger.info(f"Sparse backend: {sparse_backend_name}")

        # Initialize fusion strategy
        fusion_name = getattr(self.settings, "hybrid_fusion_strategy", "rrf")
        fusion_class = self.FUSION_STRATEGIES.get(fusion_name, ReciprocalRankFusion)
        self.fusion = fusion_class()
        logger.info(f"Fusion strategy: {fusion_name}")

        # Weights for combining scores
        self.dense_weight = getattr(self.settings, "hybrid_dense_weight", 1.0)
        self.sparse_weight = getattr(self.settings, "hybrid_sparse_weight", 1.0)

        # Retrieval depth (retrieve more than final top_k for better fusion)
        self.retrieval_k = getattr(self.settings, "hybrid_retrieval_k", 100)

    def index(self, documents: list[Document]) -> None:
        """Index documents in both dense and sparse backends"""
        logger.info(f"Indexing {len(documents)} documents in hybrid search")

        # Flatten documents to chunks for dense indexing
        all_chunks = []
        for doc in documents:
            all_chunks.extend(doc.chunks)

        # Index in both backends
        self.dense_backend.index_documents(all_chunks)
        self.sparse_backend.index_documents(documents)

        logger.info("Hybrid indexing complete")

    def search(
        self,
        query: str,
        top_k: int = None,
        query_embedding: list[float] | None = None,
        filter_fn: Callable[[RetrievedChunk], bool] | None = None,
    ) -> list[SearchResult]:
        """
        Execute hybrid search: dense + sparse → fuse → return top_k

        Args:
            query: Search query
            top_k: Final number of results (default from settings)
            query_embedding: Pre-computed query embedding (optional)
            filter_fn: Optional filter function for post-processing

        Returns:
            Fused and ranked SearchResult objects with source attribution
        """
        if top_k is None:
            top_k = getattr(self.settings, "hybrid_top_k", 10)

        try:
            # Retrieve from both backends (retrieve more than needed for better fusion)
            dense_results = self.dense_backend.search(
                query=query, query_embedding=query_embedding, top_k=self.retrieval_k
            )

            sparse_results = self.sparse_backend.search(query=query, top_k=self.retrieval_k)

            # Apply pre-fusion filter if provided
            if filter_fn:
                dense_results = [r for r in dense_results if filter_fn(r)]
                sparse_results = [r for r in sparse_results if filter_fn(r)]

            # Fuse results
            fused = self.fusion.fuse(
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
                top_k=top_k,
            )

            logger.info(
                f"Hybrid search complete: query='{query[:50]}...', " f"results={len(fused)}"
            )

            return fused

        except Exception as e:
            logger.error(f"Hybrid search error: {e}, falling back to dense only")
            # Fallback: dense retrieval only
            fallback = self.dense_backend.search(query, query_embedding, top_k)
            return [
                SearchResult(
                    chunk=chunk,
                    dense_score=chunk.score,
                    combined_score=chunk.score,
                    source="dense(fallback)",
                )
                for chunk in fallback
            ]

    def search_chunks_only(self, query: str, top_k: int = None) -> list[RetrievedChunk]:
        """
        Convenience method returning just chunks (backward compatible)
        """
        results = self.search(query, top_k)
        # Update chunk scores to combined scores
        for r in results:
            r.chunk.score = r.combined_score
        return [r.chunk for r in results]
