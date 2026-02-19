"""
Advanced RAG Pipeline Service - Production-Grade Implementation
"""

from typing import List, Dict, Any, Optional
import logging
import re
from dataclasses import dataclass
from openai import AsyncOpenAI
from app.config import settings
from app.models.retrieved_chunks import RetrievedChunk
from app.services.query_rewriting_service import QueryRewritingService
from app.services.embedding_service import EmbeddingService
from app.services.hybdrid_retrieval_service import HybridSearchService, SearchResult
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger("rag_app.advanced_rag_pipeline_service")


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    target_size: int = 350  # Target characters per chunk
    max_size: int = 500  # Maximum before forced split
    overlap: int = 50  # Character overlap between chunks
    preserve_sections: bool = True  # Keep section headers intact


@dataclass
class SectionBoundary:
    """Represents a document section boundary."""

    name: str
    start_char: int
    end_char: Optional[int] = None
    level: int = 1


class DocumentStructureAnalyzer:
    """
    Analyzes document structure to identify sections, headers, and logical boundaries.
    Runs BEFORE chunking to preserve document semantics.
    """

    SECTION_PATTERNS = [
        r"^#{1,6}\s+(.+)$",  # Markdown headers
        r"^([A-Z][A-Z\s\-]{2,50})$",  # ALL CAPS headers
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})\s*$",  # Title Case headers
        r"^(Introduction|Summary|Conclusion|Diagnosis|Treatment|Symptoms|Prevention|Prognosis)\s*$",
        r"^(What is|How to|When to|Why)\s+",  # Question-style headers
    ]

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Initial large chunks for analysis
            chunk_overlap=0,
            separators=["\n\n", "\n"],
        )

    def analyze(self, text: str) -> List[SectionBoundary]:
        """
        Identify all section boundaries in document.
        Returns ordered list of sections with character positions.
        """
        lines = text.split("\n")
        sections = []
        current_pos = 0

        for i, line in enumerate(lines):
            is_header, header_name = self._is_section_header(line)

            if is_header:
                # Close previous section
                if sections:
                    sections[-1].end_char = current_pos

                # Start new section
                sections.append(
                    SectionBoundary(
                        name=header_name, start_char=current_pos, level=self._get_header_level(line)
                    )
                )

            current_pos += len(line) + 1  # +1 for newline

        # Close final section
        if sections:
            sections[-1].end_char = len(text)

        return sections

    def _is_section_header(self, line: str) -> tuple[bool, str]:
        """Check if line is a section header."""
        line = line.strip()

        if not line or len(line) > 60:
            return False, ""

        for pattern in self.SECTION_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                # Clean up header name
                name = match.group(1) if match.groups() else line
                name = re.sub(r"^#+\s*", "", name)  # Remove markdown
                return True, name.strip()[:50]

        return False, ""

    def _get_header_level(self, line: str) -> int:
        """Determine header hierarchy level."""
        if line.startswith("######"):
            return 6
        elif line.startswith("#####"):
            return 5
        elif line.startswith("####"):
            return 4
        elif line.startswith("###"):
            return 3
        elif line.startswith("##"):
            return 2
        elif line.startswith("#"):
            return 1
        elif line.isupper():
            return 2
        return 1


class SemanticChunker:
    """
    Creates semantically coherent chunks that respect document structure.
    """

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.analyzer = DocumentStructureAnalyzer()

    def chunk(self, text: str, filename: str, base_metadata: Dict = None) -> List[RetrievedChunk]:
        """
        Create chunks that:
        1. Respect section boundaries
        2. Maintain semantic coherence (sentence boundaries)
        3. Target specific size range
        4. Preserve critical content (diagnosis criteria, etc.)
        """
        base_metadata = base_metadata or {}

        # Step 1: Analyze document structure
        sections = self.analyzer.analyze(text)
        logger.info(f"Detected {len(sections)} sections in {filename}")

        # Step 2: Create chunks within section boundaries
        chunks = []
        global_idx = 0

        for section in sections:
            section_text = text[section.start_char : section.end_char]

            # Split section into semantic units
            section_chunks = self._chunk_section(
                section_text, section, filename, global_idx, base_metadata
            )

            chunks.extend(section_chunks)
            global_idx += len(section_chunks)

        return chunks

    def _chunk_section(
        self,
        section_text: str,
        section: SectionBoundary,
        filename: str,
        start_idx: int,
        base_metadata: Dict,
    ) -> List[RetrievedChunk]:
        """Chunk a single section while preserving semantics."""

        # Protect critical passages (don't split these)
        protected_spans = self._find_protected_spans(section_text)

        # Split remaining text
        split_points = self._calculate_split_points(
            section_text, protected_spans, self.config.target_size, self.config.max_size
        )

        # Create chunks at split points
        chunks = []
        for i, (start, end) in enumerate(
            zip([0] + split_points, split_points + [len(section_text)])
        ):
            chunk_text = section_text[start:end].strip()
            if not chunk_text:
                continue

            # Determine if this chunk contains the section header
            is_header_chunk = i == 0 and section.level == 1

            chunk = RetrievedChunk(
                id=f"{filename}_{start_idx + i}",
                content=chunk_text,
                metadata={
                    **base_metadata,
                    "filename": filename,
                    "chunk_index": start_idx + i,
                    "section": section.name,
                    "section_level": section.level,
                    "is_section_start": (i == 0),
                    "is_section_header": is_header_chunk,
                    "char_start": start,
                    "char_end": end,
                },
                embedding=None,
            )
            chunks.append(chunk)

        return chunks

    def _find_protected_spans(self, text: str) -> List[tuple[int, int]]:
        """
        Find passages that should not be split (diagnosis criteria, etc.).
        """
        protected = []

        # Patterns that indicate critical content
        critical_patterns = [
            r"Your doctor will use[^.]{10,100}diabetes",
            r"blood sugar[^.]{0,50}126",
            r"Hemoglobin A1c[^.]{0,100}",
            r"You have diabetes if:",
        ]

        for pattern in critical_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Extend protection 50 chars before and after
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                protected.append((start, end))

        # Merge overlapping protections
        protected = self._merge_spans(protected)
        return protected

    def _calculate_split_points(
        self, text: str, protected_spans: List[tuple[int, int]], target_size: int, max_size: int
    ) -> List[int]:
        """Calculate optimal split points avoiding protected spans."""
        split_points = []
        current_pos = 0

        while current_pos < len(text):
            # Find next viable split point
            next_split = self._find_next_split(
                text, current_pos, target_size, max_size, protected_spans
            )

            if next_split is None:
                break

            split_points.append(next_split)
            current_pos = next_split

        return split_points

    def _find_next_split(
        self, text: str, start: int, target: int, max_size: int, protected: List[tuple[int, int]]
    ) -> Optional[int]:
        """Find next split point at sentence boundary, avoiding protected spans."""
        # Search window: target to max_size
        search_start = start + target
        search_end = min(start + max_size, len(text))

        # Find sentence boundaries in window
        candidates = []
        for match in re.finditer(r"[.!?]\s+", text[search_start:search_end]):
            pos = search_start + match.end()

            # Check if in protected span
            in_protected = any(p_start <= pos <= p_end for p_start, p_end in protected)
            if not in_protected:
                candidates.append(pos)

        # Return first good candidate, or force split if needed
        if candidates:
            return candidates[0]

        # Emergency: split at max_size if no good boundary
        if search_end < len(text):
            return search_end

        return None

    def _merge_spans(self, spans: List[tuple[int, int]]) -> List[tuple[int, int]]:
        """Merge overlapping or adjacent spans."""
        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda x: x[0])
        merged = [sorted_spans[0]]

        for current in sorted_spans[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # Overlap
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)

        return merged


class AdvancedRAGPipelineService:
    """
    Production-grade RAG pipeline with semantic chunking and structure-aware retrieval.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        hybrid_search_service: Optional[HybridSearchService] = None,
        chunking_config: Optional[ChunkingConfig] = None,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.query_rewriting_service = QueryRewritingService(api_key=self.api_key)
        self.embedding_service = EmbeddingService(api_key=self.api_key)
        self.llm_client = AsyncOpenAI(api_key=self.api_key)
        self.chunking_config = chunking_config or ChunkingConfig()
        self.chunker = SemanticChunker(self.chunking_config)

        # Hybrid search setup
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

        # Caches for neighbor lookup
        self._chunk_cache: Dict[str, RetrievedChunk] = {}
        self._chunk_index_map: Dict[str, List[str]] = {}

    async def index_documents(self, raw_chunks: List[RetrievedChunk]) -> None:
        """
        Index documents with semantic re-chunking and structure preservation.
        """
        # Step 1: Reconstruct full document text from raw chunks
        filename_to_text = self._reconstruct_documents(raw_chunks)

        # Step 2: Semantic chunking with structure analysis
        all_chunks = []
        for filename, (text, base_meta) in filename_to_text.items():
            logger.info(f"Semantic chunking: {filename}")
            chunks = self.chunker.chunk(text, filename, base_meta)
            all_chunks.extend(chunks)
            logger.info(f"  Created {len(chunks)} chunks")

        # Step 3: Generate embeddings
        chunks_need_emb = [c for c in all_chunks if c.embedding is None]
        if chunks_need_emb:
            texts = [c.content for c in chunks_need_emb]
            embeddings, _ = await self.embedding_service.generate_embeddings(texts)
            for chunk, emb in zip(chunks_need_emb, embeddings):
                chunk.embedding = emb

        # Step 4: Build caches
        self._chunk_cache = {c.id: c for c in all_chunks}
        self._chunk_index_map = self._build_index_map(all_chunks)

        # Step 5: Index to hybrid search
        documents = [
            Document(
                page_content=c.content,
                metadata={
                    "id": c.id,
                    **{k: v for k, v in c.metadata.items() if v is not None},
                    "embedding": c.embedding,
                },
                id=c.id,
            )
            for c in all_chunks
        ]

        self.hybrid_search.index(documents)
        logger.info(f"Indexed {len(documents)} semantic chunks")

    def _reconstruct_documents(
        self, raw_chunks: List[RetrievedChunk]
    ) -> Dict[str, tuple[str, Dict]]:
        """
        Reconstruct full document text from raw chunks.
        Returns: {filename: (full_text, base_metadata)}
        """
        # Group by filename
        filename_chunks: Dict[str, List[RetrievedChunk]] = {}
        for chunk in raw_chunks:
            fname = chunk.metadata.get("filename", "unknown")
            if fname not in filename_chunks:
                filename_chunks[fname] = []
            filename_chunks[fname].append(chunk)

        # Sort by chunk_index and reconstruct
        result = {}
        for fname, chunks in filename_chunks.items():
            sorted_chunks = sorted(chunks, key=lambda c: c.metadata.get("chunk_index", 0))
            full_text = "\n".join(c.content for c in sorted_chunks)
            base_meta = {
                k: v
                for k, v in sorted_chunks[0].metadata.items()
                if k not in ["chunk_index", "char_start", "char_end"]
            }
            result[fname] = (full_text, base_meta)

        return result

    def _build_index_map(self, chunks: List[RetrievedChunk]) -> Dict[str, List[str]]:
        """Build filename -> ordered chunk IDs mapping."""
        index_map: Dict[str, List[str]] = {}
        for chunk in sorted(chunks, key=lambda c: c.metadata.get("chunk_index", 0)):
            fname = chunk.metadata.get("filename", "unknown")
            if fname not in index_map:
                index_map[fname] = []
            index_map[fname].append(chunk.id)
        return index_map

    async def answer_query(
        self,
        query: str,
        top_k: int = 5,
        use_rewrites: bool = True,
        use_expansions: bool = True,
        max_rewrites: int = 3,
        max_expansions: int = 5,
        expand_neighbors: bool = True,
        neighbor_window: int = 1,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline with semantic retrieval and context expansion.
        """
        if not query or not query.strip():
            return self._empty_response(query)

        # Generate query variants
        variants = await self._generate_query_variants(
            query, use_rewrites, use_expansions, max_rewrites, max_expansions
        )

        # Get embeddings
        embeddings = await self._embed_query_variants(variants)

        # Retrieve with section-aware boosting
        results = await self._retrieve_with_boost(query, variants, embeddings, top_k * 2)

        # Expand with neighbors
        if expand_neighbors:
            results = self._expand_with_neighbors(results, neighbor_window)

        # Final ranking and deduplication
        final_results = self._deduplicate_and_rank(results)[:top_k]

        # Build context and generate answer
        context = self._build_context(final_results)

        if not context.strip():
            return self._empty_response(query, final_results)

        answer = await self._generate_answer(query, context)

        return {
            "question": query,
            "answer": answer,
            "sources": self._format_sources(final_results),
            "chunks_used": len(final_results),
            "model": self.model,
            "retrieval_stats": {
                "query_variants": len(variants),
                "unique_results": len(set(r.chunk.id for r in final_results)),
                "final_results": len(final_results),
            },
        }

    def _deduplicate_and_rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Deduplicate by chunk ID and sort by combined score (descending).
        Keeps the highest scoring result for each chunk.
        """
        seen: Dict[str, SearchResult] = {}

        for result in results:
            cid = result.chunk.id
            if not cid:
                continue

            # Keep highest scoring version of each chunk
            if cid not in seen or result.combined_score > seen[cid].combined_score:
                seen[cid] = result

        # Sort by score descending
        return sorted(seen.values(), key=lambda x: x.combined_score, reverse=True)

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

    async def _retrieve_with_boost(
        self,
        original_query: str,
        variants: List[str],
        embeddings: List[List[float]],
        retrieve_k: int,
    ) -> List[SearchResult]:
        """
        Multi-variant retrieval with section-based boosting.
        """
        all_results = []
        query_lower = original_query.lower()

        # Detect query intent
        target_section = None
        section_keywords = {
            "diagnosis": ["diagnos", "test", "criteria", "126", "fasting", "blood sugar"],
            "treatment": ["treat", "medication", "medicine", "drug", "pill", "insulin"],
            "symptoms": ["symptom", "sign", "feel", "urination", "thirst", "hunger"],
        }
        for section, keywords in section_keywords.items():
            if any(kw in query_lower for kw in keywords):
                target_section = section
                break

        # Retrieve for all variants
        for variant, emb in zip(variants, embeddings):
            variant_results = await self.hybrid_search.search(
                query=variant,
                query_embedding=emb,
                top_k=retrieve_k,
            )
            all_results.extend(variant_results)

        # Apply section boosting
        if target_section:
            for r in all_results:
                chunk_section = r.chunk.metadata.get("section", "").lower()

                if target_section in chunk_section:
                    r.combined_score *= 2.0  # 100% boost
                    r.source = f"{r.source}+section_match"
                elif self._is_irrelevant_section(chunk_section, target_section):
                    r.combined_score *= 0.3  # 70% penalty

        return all_results

    def _is_irrelevant_section(self, chunk_section: str, target: str) -> bool:
        """Check if chunk section is clearly irrelevant to target."""
        irrelevant_map = {
            "diagnosis": ["treatment", "medication", "prevention", "prognosis"],
            "treatment": ["diagnosis", "symptoms", "prevention"],
            "symptoms": ["treatment", "medication", "prevention"],
        }
        irrelevant = irrelevant_map.get(target, [])
        return any(inv in chunk_section for inv in irrelevant)

    def _expand_with_neighbors(
        self,
        results: List[SearchResult],
        window: int,
    ) -> List[SearchResult]:
        """Expand results with adjacent chunks."""
        expanded = []
        seen_ids = set()

        for result in results:
            # Add main result
            if result.chunk.id not in seen_ids:
                expanded.append(result)
                seen_ids.add(result.chunk.id)

            # Get neighbors
            neighbors = self._get_neighbor_chunks(result.chunk, window)

            for neighbor in neighbors:
                if neighbor.id in seen_ids:
                    continue

                neighbor_result = SearchResult(
                    chunk=neighbor,
                    dense_score=0.0,
                    sparse_score=0.0,
                    combined_score=result.combined_score * 0.6,
                    source="neighbor",
                )
                expanded.append(neighbor_result)
                seen_ids.add(neighbor.id)

        return expanded

    def _get_neighbor_chunks(
        self,
        chunk: RetrievedChunk,
        window: int,
    ) -> List[RetrievedChunk]:
        """Get adjacent chunks from cache."""
        fname = chunk.metadata.get("filename")
        idx = chunk.metadata.get("chunk_index")

        if fname is None or idx is None:
            return []

        file_chunks = self._chunk_index_map.get(fname, [])
        if not file_chunks:
            return []

        # Find position
        try:
            pos = file_chunks.index(chunk.id)
        except ValueError:
            return []

        # Get neighbors
        neighbors = []
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            n_pos = pos + offset
            if 0 <= n_pos < len(file_chunks):
                n_id = file_chunks[n_pos]
                neighbor = self._chunk_cache.get(n_id)
                if neighbor:
                    neighbors.append(neighbor)

        return neighbors

    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context string with proper formatting."""
        parts = []

        for i, r in enumerate(results, 1):
            chunk = r.chunk
            meta = chunk.metadata or {}

            # Format header
            section = meta.get("section", "Unknown")
            is_neighbor = "neighbor" in r.source

            header = (
                f"[{i}] {meta.get('filename', 'Unknown')} | "
                f"Section: {section} | "
                f"Score: {r.combined_score:.4f} | "
                f"Source: {r.source}"
            )

            if is_neighbor:
                header += " [CONTEXT]"

            # Truncate if needed
            text = chunk.content
            if len(text) > 800 and not is_neighbor:
                text = text[:800] + "... [truncated]"

            parts.append(f"{header}\n{text}")

        return "\n\n".join(parts)

    def _format_sources(self, results: List[SearchResult]) -> List[Dict]:
        """Format results as source entries."""
        return [
            {
                "id": r.chunk.id,
                "filename": r.chunk.metadata.get("filename", ""),
                "chunk_index": r.chunk.metadata.get("chunk_index", 0),
                "section": r.chunk.metadata.get("section", ""),
                "score": r.combined_score,
                "source_type": r.source,
                "dense_score": r.dense_score,
                "sparse_score": r.sparse_score,
                "text": (
                    (r.chunk.content[:200] + "...")
                    if len(r.chunk.content) > 200
                    else r.chunk.content
                ),
                "is_neighbor": "neighbor" in r.source,
            }
            for r in results
        ]

    # ... other helper methods (_generate_query_variants, _embed_query_variants, etc.) ...

    def _empty_response(self, query: str, results: List[SearchResult] = None) -> Dict:
        """Return empty response template."""
        return {
            "question": query,
            "answer": "I don't have enough information to answer that question.",
            "sources": self._format_sources(results) if results else [],
            "chunks_used": 0,
            "model": self.model,
            "retrieval_stats": {"query_variants": 0, "unique_results": 0, "final_results": 0},
        }

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer from context."""
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer based only on the context above. "
            "If the exact answer is not present, provide the closest supported information "
            "and clearly state what is missing."
        )

        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers based on provided context only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content
