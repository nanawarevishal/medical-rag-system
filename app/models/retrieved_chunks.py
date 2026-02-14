from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from datetime import datetime


class RetrievedChunk(BaseModel):
    """
    A chunk of text retrieved from the vector database or search index.
    Passes through retrieval → reranking → generation pipeline.
    """

    # Required fields
    id: str = Field(..., description="Unique chunk ID (often doc_id_chunk_idx)")
    content: str = Field(..., description="The actual text content")

    # Source tracking
    document_id: Optional[str] = Field(None, description="Parent document ID")
    chunk_index: Optional[int] = Field(None, description="Position in parent doc")

    # Scoring (updated at each stage)
    score: float = Field(0.0, description="Retrieval/relevance score")
    vector_score: Optional[float] = Field(None, description="Dense retrieval score")
    keyword_score: Optional[float] = Field(None, description="BM25/keyword score")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")

    # Optional context
    title: Optional[str] = Field(None, description="Document title")
    source: Optional[str] = Field(None, description="File path or URL")
    created_at: Optional[datetime] = Field(None, description="Chunk creation time")

    def __repr__(self):
        preview = self.content[:50].replace("\n", " ")
        return f"Chunk({self.id}, score={self.score:.3f}, '{preview}...')"
