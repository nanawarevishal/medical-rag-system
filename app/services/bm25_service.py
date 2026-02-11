"""
BM25 (Sparse) Retrieval Service
Builds a sparse index from cached chunks and performs BM25 search.
"""

from typing import List, Dict, Any, Tuple
import logging
import math
import re
from app.services.cache_service import CacheService

logger = logging.getLogger("rag_app.bm25_service")


class BM25Service:
    """Simple BM25 implementation over cached chunks."""

    DEFAULT_EXTENSIONS = ["pdf", "txt", "md", "markdown", "docx", "doc", "csv", "json"]

    def __init__(self, cache_service: CacheService | None = None):
        """
        Initialize BM25 service.

        Args:
            cache_service: Optional CacheService for loading cached chunks
        """
        self.cache_service = cache_service or CacheService()
        self._corpus: List[Dict[str, Any]] = []
        self._doc_freq: Dict[str, int] = {}
        self._term_freqs: List[Dict[str, int]] = []
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0

    # ==================== Public API ====================

    def build_index(self) -> Dict[str, Any]:
        """
        Build BM25 index from cached chunks.

        Returns:
            Dictionary with index stats
        """
        chunks = self._load_all_chunks()
        self._build_bm25(chunks)

        return {
            "documents_indexed": len(self._corpus),
            "avg_doc_length": round(self._avgdl, 2),
        }

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform BM25 search over cached chunks.

        Args:
            query: User query
            top_k: Number of results to return

        Returns:
            List of chunks with BM25 score
        """
        if not query or not query.strip():
            return []

        if not self._corpus:
            self.build_index()

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._score(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            chunk = {**self._corpus[idx], "sparse_score": float(score)}
            results.append(chunk)
        return results

    # ==================== Internal Helpers ====================

    def _load_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Load all cached chunks from storage backend.

        Returns:
            List of chunk dicts
        """
        chunks: List[Dict[str, Any]] = []
        storage = self.cache_service.storage
        document_ids = storage.list_documents()

        if not document_ids:
            logger.warning("BM25: No cached documents found to index")
            return []

        for doc_id in document_ids:
            file_extension = self._guess_extension(doc_id)
            if not file_extension:
                logger.debug(f"BM25: Could not determine extension for {doc_id}")
                continue

            try:
                doc_chunks = storage.load_chunks(doc_id, file_extension)
            except Exception as e:
                logger.warning(f"BM25: Failed to load chunks for {doc_id}: {e}")
                continue

            for chunk in doc_chunks:
                # Ensure stable chunk id
                chunk_id = chunk.get("id")
                if not chunk_id:
                    chunk_index = chunk.get("chunk_index", 0)
                    chunk_id = f"{doc_id}_{chunk_index}"

                chunks.append({
                    "id": chunk_id,
                    "text": chunk.get("text", ""),
                    "metadata": {
                        "filename": chunk.get("filename", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "token_count": chunk.get("token_count", 0),
                        "doc_id": doc_id,
                        "file_extension": file_extension
                    }
                })

        logger.info(f"BM25: Loaded {len(chunks)} chunks for indexing")
        return chunks

    def _guess_extension(self, doc_id: str) -> str | None:
        """
        Guess file extension for a document by checking existence.
        Required for S3 where folders are keyed by extension.
        """
        for ext in self.DEFAULT_EXTENSIONS:
            try:
                if self.cache_service.storage.exists(doc_id, ext):
                    return ext
            except Exception:
                continue
        return None

    def _build_bm25(self, chunks: List[Dict[str, Any]]):
        self._corpus = chunks
        self._term_freqs = []
        self._doc_freq = {}
        self._doc_len = []

        for chunk in self._corpus:
            tokens = self._tokenize(chunk.get("text", ""))
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._term_freqs.append(tf)
            self._doc_len.append(len(tokens))

            for term in tf.keys():
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

        self._avgdl = sum(self._doc_len) / max(len(self._doc_len), 1)

    def _score(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
        N = len(self._corpus)
        scores = [0.0] * N

        for i, tf in enumerate(self._term_freqs):
            dl = self._doc_len[i] or 1
            score = 0.0
            for term in query_tokens:
                if term not in tf:
                    continue
                df = self._doc_freq.get(term, 0)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                freq = tf[term]
                denom = freq + k1 * (1 - b + b * (dl / (self._avgdl or 1)))
                score += idf * ((freq * (k1 + 1)) / denom)
            scores[i] = score
        return scores

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        return tokens
