"""
Multi-Source RAG + Text-to-SQL API
FastAPI application with document RAG and natural language to SQL capabilities.
"""

from typing import Optional, Dict, Any
from fastapi import FastAPI, status, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import sys
import shutil

from app.config import settings
from app.logging_config import setup_logging, get_logger
from app.models.retrieved_chunks import RetrievedChunk
from app.services.document_service import parse_document, chunk_text
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.rag_service import RAGService
from app.services.advanced_rag_pipeline_service import (
    AdvancedRAGPipelineService,
    HybridSearchService,
)
from app.utils import (
    FileValidator,
    QueryValidator,
    ValidationError,
    ErrorResponse,
    format_file_size,
    truncate_text,
)

# Initialize logging
logger = setup_logging(log_level="INFO")

# OPIK monitoring (optional - gracefully handles if not configured)
try:
    from opik import track

    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False

    # Create a no-op decorator if OPIK is not installed
    def track(name=None, **kwargs):
        def decorator(func):
            return func

        return decorator


app = FastAPI(
    title="Multi-Source RAG + Text-to-SQL API",
    description="A system that combines document RAG with natural language to SQL conversion",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path=settings.ROOT_PATH,
)

# Global service instances (initialized on startup if API keys are available)
embedding_service: EmbeddingService | None = None
vector_service: VectorService | None = None
rag_service: RAGService | None = None
advanced_rag_service: AdvancedRAGPipelineService | None = None

# Upload directory (from config, supports both Lambda /tmp and local paths)
UPLOAD_DIR = Path(settings.UPLOAD_DIR)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify the API is running and check service connectivity.
    """
    # Check service availability
    services_status = {
        "embedding_service": embedding_service is not None,
        "vector_service": vector_service is not None,
        "rag_service": rag_service is not None,
        "advanced_rag_service": advanced_rag_service is not None,
    }

    # Determine overall health
    any_service_available = any(services_status.values())
    health_status = "healthy" if any_service_available else "degraded"

    return {
        "status": health_status,
        "service": "Multi-Source RAG + Text-to-SQL API",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "services": services_status,
        "features_available": {
            "document_rag": services_status["rag_service"],
            "advanced_rag_hybrid": services_status["advanced_rag_service"],
            "text_to_sql": services_status["sql_service"],
        },
        "configuration": {
            "openai_configured": settings.OPENAI_API_KEY is not None,
            "pinecone_configured": settings.PINECONE_API_KEY is not None,
            "database_configured": settings.DATABASE_URL is not None,
        },
    }


@app.get("/info", status_code=status.HTTP_200_OK, tags=["Information"])
async def get_info():
    """
    Get system information and configuration details.
    """
    return {
        "application": {
            "name": "Multi-Source RAG + Text-to-SQL",
            "version": "0.1.0",
            "environment": "development",
        },
        "features": {
            "document_rag": "Available - Basic RAG",
            "advanced_rag_hybrid": "Available - Hybrid Search (BM25 + Dense + RRF)",
            "text_to_sql": "Available",
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "upload_document": "POST /upload",
            "pipeline_query": "POST /query/pipeline (Hybrid Search)",
        },
    }


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with welcome message and quick links.
    """
    return {
        "message": "Welcome to Multi-Source RAG + Text-to-SQL API",
        "version": "0.1.0",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.post("/upload", status_code=status.HTTP_201_CREATED, tags=["Documents"])
@track(name="upload_document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, DOCX, CSV, JSON, TXT).
    Pipeline: validate → save → parse → chunk → embed → store
    """
    global embedding_service, vector_service

    # Validate file
    try:
        FileValidator.validate_file(file)
    except ValidationError as e:
        raise HTTPException(
            status_code=400, detail=ErrorResponse.validation_error(str(e), field="file")
        )

    # Check if services are initialized
    if not embedding_service or not vector_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable(
                "Document RAG services",
                "Please configure OPENAI_API_KEY and PINECONE_API_KEY in .env",
            ),
        )

    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_extension = file_path.suffix.lstrip(".").lower()

        # Parse and chunk with context-aware approach
        logger.info(f"Parsing and chunking document: {file.filename}")
        from app.services.document_service import parse_and_chunk_with_context

        chunks = parse_and_chunk_with_context(
            str(file_path),
            chunk_size=settings.CHUNK_SIZE,
            min_chunk_size=settings.MIN_CHUNK_SIZE,
        )
        logger.info(f"Created {len(chunks)} context-aware chunks")

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings, _ = await embedding_service.generate_embeddings(texts)

        # Store in Pinecone
        # logger.info(f"Storing {len(chunks)} vectors in Pinecone...")
        # vector_service.add_documents(
        #     chunks=chunks, embeddings=embeddings, filename=file.filename, namespace="default"
        # )

        # Index in hybrid search if available
        if advanced_rag_service:
            # Create chunks with embeddings attached
            hybrid_chunks = [
                RetrievedChunk(
                    id=f"{file.filename}_{i}",
                    content=chunk["text"],
                    metadata={"filename": file.filename, "chunk_index": i, **chunk},
                    embedding=emb,  # CRITICAL: Attach embedding here
                )
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ]

            # This indexes to BOTH Pinecone (dense) and BM25 (sparse)
            await advanced_rag_service.index_documents(hybrid_chunks)

        file_size = file_path.stat().st_size
        total_tokens = sum(chunk["token_count"] for chunk in chunks)

        return {
            "status": "success",
            "filename": file.filename,
            "file_size": format_file_size(file_size),
            "chunks_created": len(chunks),
            "total_tokens": total_tokens,
            "message": f"Document processed and {len(chunks)} chunks stored",
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=ErrorResponse.validation_error(str(e)))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=ErrorResponse.internal_error("upload document", e)
        )


@app.get("/documents", status_code=status.HTTP_200_OK, tags=["Documents"])
async def list_documents():
    """
    List all uploaded documents.
    """
    try:
        documents = []
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                documents.append(
                    {
                        "filename": file_path.name,
                        "size_bytes": file_path.stat().st_size,
                        "uploaded_at": datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ).isoformat(),
                    }
                )

        return {"total_documents": len(documents), "documents": documents}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/stats", status_code=status.HTTP_200_OK, tags=["Information"])
async def get_stats():
    """
    Get system statistics and usage information.
    """
    try:
        # Count uploaded documents
        documents = []
        total_size = 0

        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                file_size = file_path.stat().st_size
                documents.append(
                    {
                        "filename": file_path.name,
                        "size_bytes": file_size,
                    }
                )
                total_size += file_size

        return {
            "documents": {
                "total_uploaded": len(documents),
                "total_size": format_file_size(total_size),
                "total_size_bytes": total_size,
            },
            "services": {
                "embedding": embedding_service is not None,
                "vector_db": vector_service is not None,
                "advanced_rag": advanced_rag_service is not None,
                "sql": sql_service is not None,
            },
            "configuration": {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "max_file_size": format_file_size(FileValidator.MAX_FILE_SIZE),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=ErrorResponse.internal_error("get statistics", e)
        )


@app.delete("/vectors/clear", status_code=status.HTTP_200_OK, tags=["Vectors"])
async def clear_vectors(namespace: Optional[str] = "default", confirm: bool = False):
    """
    Clear all vectors from the Pinecone vector database.
    """
    global vector_service

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Confirmation required",
                "message": "You must set confirm=true to clear vectors.",
                "example": "/vectors/clear?namespace=default&confirm=true",
            },
        )

    if not vector_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable("Vector database not initialized."),
        )

    try:
        stats_before = vector_service.get_index_stats()
        total_vectors_before = stats_before.get("total_vector_count", 0)

        result = vector_service.delete_all_vectors(namespace=namespace)

        stats_after = vector_service.get_index_stats()
        total_vectors_after = stats_after.get("total_vector_count", 0)

        return {
            "status": result["status"],
            "namespaces_cleared": result["namespaces_cleared"],
            "vector_count_before": total_vectors_before,
            "vector_count_after": total_vectors_after,
            "vectors_deleted": total_vectors_before - total_vectors_after,
            "message": result["message"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=ErrorResponse.internal_error("clear_vectors", e)
        )


class PipelineQueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_rewrites: bool = True
    use_expansions: bool = True
    max_rewrites: int = 3
    max_expansions: int = 5
    sparse_weight: float = 1.0
    dense_weight: float = 1.0


@app.post("/query/pipeline", status_code=status.HTTP_200_OK, tags=["Query"])
@track(name="pipeline_query")
async def pipeline_query(payload: PipelineQueryRequest):
    """
    Pipeline query endpoint using Hybrid Search (BM25 + Dense + RRF).
    """
    global advanced_rag_service

    if not advanced_rag_service:
        raise HTTPException(
            status_code=503,
            detail="Advanced RAG pipeline not initialized. Please configure OPENAI_API_KEY in .env file.",
        )

    try:
        result = await advanced_rag_service.answer_query(
            query=payload.question,
            top_k=payload.top_k,
            use_rewrites=payload.use_rewrites,
            use_expansions=payload.use_expansions,
            max_rewrites=payload.max_rewrites,
            max_expansions=payload.max_expansions,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline query failed: {str(e)}")


def initialize_services():
    """Initialize all services. Called on startup."""
    global embedding_service, vector_service, rag_service, advanced_rag_service, sql_service

    # Ensure upload directory exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory initialized: {UPLOAD_DIR}")

    logger.info("=" * 60)
    logger.info("Starting Multi-Source RAG + Text-to-SQL API...")
    logger.info("=" * 60)

    try:
        # 1. Initialize Embedding Service (required for RAG)
        if settings.OPENAI_API_KEY:
            logger.info("Initializing Embedding Service...")
            embedding_service = EmbeddingService(api_key=settings.OPENAI_API_KEY)
            logger.info("✓ Embedding Service initialized")
        else:
            logger.warning("OPENAI_API_KEY not set. Embedding Service not initialized.")

        # 2. Initialize Vector Service (Pinecone)
        if settings.PINECONE_API_KEY and embedding_service:
            logger.info("Initializing Vector Service...")
            vector_service = VectorService()
            logger.info("✓ Vector Service initialized")
        else:
            logger.warning("PINECONE_API_KEY not set. Vector Service not initialized.")

        # 3. Initialize Basic RAG Service
        if embedding_service and vector_service:
            logger.info("Initializing Basic RAG Service...")
            rag_service = RAGService()
            logger.info("✓ Basic RAG Service initialized")

        # 4. Initialize Advanced RAG Pipeline Service (Hybrid Search)
        if embedding_service and settings.OPENAI_API_KEY:
            logger.info("Initializing Advanced RAG Pipeline Service with Hybrid Search...")

            # Create hybrid search service
            hybrid_search = HybridSearchService()

            # Create advanced RAG pipeline
            advanced_rag_service = AdvancedRAGPipelineService(
                api_key=settings.OPENAI_API_KEY, hybrid_search_service=hybrid_search
            )
            logger.info("✓ Advanced RAG Pipeline initialized (BM25 + Dense + RRF)")

    except Exception as e:
        logger.error(f"Service initialization error: {e}")

    logger.info("=" * 60)
    logger.info("Service Status:")
    logger.info(f"  - Embedding: {'✓' if embedding_service else '✗'}")
    logger.info(f"  - Vector DB: {'✓' if vector_service else '✗'}")
    logger.info(f"  - Basic RAG: {'✓' if rag_service else '✗'}")
    logger.info(f"  - Advanced RAG (Hybrid): {'✓' if advanced_rag_service else '✗'}")
    logger.info("=" * 60)


@app.on_event("startup")
async def startup_event():
    """Execute tasks on application startup."""
    initialize_services()


@app.on_event("shutdown")
async def shutdown_event():
    """Execute cleanup tasks on application shutdown."""
    logger.info("Shutting down...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
