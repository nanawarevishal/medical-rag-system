# Multi-Level Cache Architecture

## Overview
Comprehensive caching strategy showing Redis query cache (Tier 1) and S3/Local document cache (Tier 2) working together to optimize performance and reduce API costs.

## Architecture Diagram

```mermaid
graph TB
    Request[Client Request]

    subgraph "Tier 1: Redis Query Cache - Hot Data"
        Redis[Upstash Redis Cloud<br/>5-10ms retrieval<br/>Serverless, pay-per-request]

        subgraph "RAG Query Caches"
            RAGCache[RAG Answers<br/>TTL: 1 hour<br/>Key: rag:hash<br/>Size: ~2KB/entry]
            EmbedCache[Question Embeddings<br/>TTL: 7 days<br/>Key: embedding:hash<br/>Size: ~6KB/entry]
        end

        subgraph "SQL Query Caches"
            SQLGenCache[SQL Generation<br/>TTL: 24 hours<br/>Key: sql_gen:hash<br/>Size: ~500B/entry]
            SQLResultCache[SQL Results<br/>TTL: 15 minutes<br/>Key: sql_result:hash<br/>Size: varies]
        end

        Redis --> RAGCache
        Redis --> EmbedCache
        Redis --> SQLGenCache
        Redis --> SQLResultCache
    end

    subgraph "Tier 2: Document Cache - Warm Data"
        S3[AWS S3 / Local Storage<br/>100-200ms retrieval<br/>Content-based deduplication]

        Chunks[chunks.json<br/>Text chunks + metadata<br/>~10-50KB per document]
        Embeddings[embeddings.npy<br/>NumPy vectors<br/>~100-500KB per document]
        Metadata[metadata.json<br/>Document info<br/>~1-2KB per document]

        S3 --> Chunks
        S3 --> Embeddings
        S3 --> Metadata
    end

    subgraph "Tier 3: Full Processing - Cold Path"
        OpenAI[OpenAI API<br/>2-5s processing<br/>$15-30 per 1K queries]
        Pinecone[Pinecone Vector DB<br/>50-100ms search<br/>Vector operations]
        Supabase[Supabase PostgreSQL<br/>50-200ms query<br/>SQL execution]
    end

    Request --> Redis
    Redis -.Cache MISS.-> S3
    S3 -.Cache MISS.-> OpenAI
    S3 -.Cache MISS.-> Pinecone
    Redis -.Cache MISS.-> Supabase

    OpenAI -.Store.-> Redis
    Pinecone -.Store.-> S3
    Supabase -.Store.-> Redis

    style Redis fill:#e1f5ff
    style S3 fill:#ffe1e1
    style RAGCache fill:#e1ffe1
    style EmbedCache fill:#e1ffe1
    style SQLGenCache fill:#e1ffe1
    style SQLResultCache fill:#e1ffe1
```

## Tier 1: Redis Query Cache

### Purpose
Fast, in-memory caching for frequently accessed queries and computed results.

### Cache Types

#### 1. RAG Answer Cache
**Key Pattern**: `rag:{sha256(question + top_k)}`

**Stored Data:**
```json
{
  "answer": "Based on the documents, our return policy...",
  "sources": [...],
  "chunks_retrieved": 3,
  "timestamp": "2024-01-15T10:30:00Z",
  "tokens_used": 1234
}
```

**Configuration:**
- **TTL**: 1 hour (3600 seconds)
- **Invalidation**: When new documents uploaded (entire `rag:*` pattern deleted)
- **Size**: ~2KB per cached answer
- **Hit Rate**: 40-60%

**Benefits:**
- 2-5s → 5-10ms (99% latency reduction)
- $15-30 per 1K → $0.00 (100% cost savings)
- Consistent answers for identical questions

#### 2. Embedding Cache
**Key Pattern**: `embedding:{sha256(text)}`

**Stored Data:**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...],  // 1536 floats
  "model": "text-embedding-3-small",
  "timestamp": "2024-01-15T10:30:00Z",
  "usage": {"total_tokens": 45}
}
```

**Configuration:**
- **TTL**: 7 days (604800 seconds)
- **Invalidation**: Natural expiration only
- **Size**: ~6KB per cached embedding
- **Hit Rate**: 60% (common questions + document chunks)

**Benefits:**
- 200-500ms → 5-10ms (95% latency reduction)
- $0.13 per 1K → $0.00 (100% cost savings)
- Shared across RAG queries and document uploads

#### 3. SQL Generation Cache
**Key Pattern**: `sql_gen:{sha256(question)}`

**Stored Data:**
```json
{
  "sql": "SELECT COUNT(*) FROM customers WHERE created_at >= '2024-01-01'",
  "explanation": "This query counts all customers created in 2024...",
  "confidence": 0.95,
  "timestamp": "2024-01-15T10:30:00Z",
  "tokens_used": 567
}
```

**Configuration:**
- **TTL**: 24 hours (86400 seconds)
- **Invalidation**: Natural expiration (schema changes would need manual clear)
- **Size**: ~500B per cached SQL
- **Hit Rate**: 60% (common analytics queries)

**Benefits:**
- 2-5s → 5-10ms (99% latency reduction)
- $0.20 per query → $0.00 (100% cost savings)
- Consistent SQL for same business questions

#### 4. SQL Result Cache
**Key Pattern**: `sql_result:{sha256(sql_query)}`

**Stored Data:**
```json
{
  "rows": [...],  // Query results
  "columns": ["id", "name", "total"],
  "row_count": 10,
  "execution_time_ms": 145,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Configuration:**
- **TTL**: 15 minutes (900 seconds)
- **Invalidation**: Natural expiration (data changes frequently)
- **Size**: Varies (10B - 10MB depending on result size)
- **Hit Rate**: 40% (dashboards, repeated analytics)

**Benefits:**
- 50-200ms → 5-10ms (90% latency reduction)
- Reduces database load
- Faster dashboard rendering

**Note**: Shorter TTL than SQL generation because underlying data changes more frequently than schema/questions.

### Redis Configuration

**Provider**: Upstash Redis (Serverless)

**Connection:**
```python
REDIS_URL = "rediss://default:password@region.upstash.io:6379"
```

**Features:**
- Pay-per-request pricing (cost-effective for variable load)
- REST API fallback (works in Lambda cold starts)
- TLS encryption
- Global replication (low latency worldwide)

**Cost Optimization:**
- Only cache high-value operations (> 1s processing time)
- Use appropriate TTLs (shorter for volatile data)
- Pattern-based deletion for bulk invalidation

## Tier 2: Document Cache (S3/Local)

### Purpose
Intermediate storage for parsed documents to avoid re-processing identical files.

### Storage Structure

```
documents/
  {sha256_hash}/
    chunks.json           # Parsed and chunked text
    embeddings.npy        # Pre-computed embedding vectors
    metadata.json         # Document metadata and timestamps
```

### Cache Files

#### chunks.json
```json
{
  "chunks": [
    {
      "text": "Our return policy allows customers to return...",
      "chunk_index": 0,
      "token_count": 342,
      "metadata": {
        "source_page": 1,
        "section": "Returns"
      }
    },
    ...
  ],
  "total_chunks": 45,
  "chunk_size": 512,
  "overlap": 50
}
```

**Size**: ~10-50KB per document (depends on length)

#### embeddings.npy
Binary NumPy array: `shape=(num_chunks, 1536), dtype=float32`

**Size**: `num_chunks * 1536 * 4 bytes`
- 45 chunks = ~270KB
- 100 chunks = ~600KB

#### metadata.json
```json
{
  "filename": "pricing_policy.pdf",
  "sha256": "abc123...",
  "file_size_bytes": 5242880,
  "content_type": "application/pdf",
  "uploaded_at": "2024-01-15T10:30:00Z",
  "processed_at": "2024-01-15T10:30:02Z",
  "processing_time_ms": 2340,
  "parser": "docling",
  "chunk_count": 45
}
```

**Size**: ~1-2KB

### S3 Backend Configuration

**Bucket Structure:**
```
s3://my-rag-bucket/
  documents/
    abc123def456.../  # SHA-256 hash as folder name
      chunks.json
      embeddings.npy
      metadata.json
```

**Access Pattern:**
- Upload: ~100-200ms (1-5 MB files)
- Download: ~100-200ms (same)
- List: ~50ms (check existence)

**Cost:**
- Storage: $0.023 per GB/month
- GET requests: $0.0004 per 1K requests
- PUT requests: $0.005 per 1K requests

### Local Fallback

**Path**: `./document_cache/`

**When Used:**
- Development environment
- AWS credentials not configured
- S3 access issues

**Performance:**
- Read: 10-50ms (local SSD)
- Write: 10-50ms
- No network latency

## Cache Flow Patterns

### Pattern 1: RAG Query with Full Cache Hit
```mermaid
sequenceDiagram
    Client->>Redis: Check RAG cache (rag:hash)
    Redis-->>Client: CACHE HIT (5ms)
    Client->>Client: Return cached answer
    Note over Client: Total: 5-10ms, $0.00
```

### Pattern 2: RAG Query with Partial Cache (Embedding Hit)
```mermaid
sequenceDiagram
    Client->>Redis: Check RAG cache (rag:hash)
    Redis-->>Client: CACHE MISS
    Client->>Redis: Check embedding cache (embedding:hash)
    Redis-->>Client: CACHE HIT (7ms)
    Client->>Pinecone: Vector search (80ms)
    Pinecone-->>Client: Top-3 chunks
    Client->>OpenAI: Generate answer (3.2s)
    OpenAI-->>Client: Answer
    Client->>Redis: Store RAG response (TTL: 1h)
    Note over Client: Total: 3.3s, $18 per 1K
```

### Pattern 3: Document Upload with Cache Hit
```mermaid
sequenceDiagram
    Client->>Client: Compute SHA-256 hash
    Client->>S3: Check existence (documents/hash/)
    S3-->>Client: Found (120ms)
    Client->>S3: Download chunks + embeddings (180ms)
    Client->>Pinecone: Upsert vectors (200ms)
    Client->>Redis: DELETE rag:* (invalidate)
    Note over Client: Total: 500ms, $0.00
```

## Performance Comparison

| Scenario | No Cache | Tier 1 Only | Tier 1 + Tier 2 | Savings |
|----------|----------|-------------|-----------------|---------|
| **RAG Query** | 2.5-5s | 5-10ms | 5-10ms | 99% |
| **SQL Query** | 2.5-5s | 5-10ms | 5-10ms | 99% |
| **Document Upload** | 2-6s | 2-6s | 200-500ms | 75% |
| **Embedding Generation** | 200-500ms | 5-10ms | 5-10ms | 98% |

## Cost Analysis (Per 1,000 Queries)

| Component | No Cache | With Cache | Savings |
|-----------|----------|------------|---------|
| OpenAI GPT-4 | $20.00 | $8.00 | $12.00 (60% hit rate) |
| OpenAI Embeddings | $0.13 | $0.05 | $0.08 (60% hit rate) |
| Pinecone Queries | $0.10 | $0.10 | $0.00 (included) |
| Redis Operations | $0.00 | $0.01 | -$0.01 |
| S3 Operations | $0.00 | $0.01 | -$0.01 |
| **Total** | **$20.23** | **$8.17** | **$12.06 (60% savings)** |

## Cache Invalidation Strategies

### Automatic Invalidation

**Trigger**: New document uploaded

**Action**:
```python
# Delete all RAG response cache keys
redis_client.delete_pattern("rag:*")
```

**Reason**: New documents may contain answers to previously cached queries

### Manual Invalidation

**Endpoint**: `DELETE /cache/clear?cache_type=X`

**Options:**
- `cache_type=all` → FLUSHDB (clear everything)
- `cache_type=rag` → DELETE `rag:*`
- `cache_type=embeddings` → DELETE `embedding:*`
- `cache_type=sql_gen` → DELETE `sql_gen:*`
- `cache_type=sql_result` → DELETE `sql_result:*`
- `cache_type=documents` → Clear S3/local + DELETE `doc:*`

### Pattern-Based Deletion

```python
def clear_cache_by_pattern(pattern: str):
    cursor = 0
    while True:
        cursor, keys = redis_client.scan(cursor, match=pattern, count=100)
        if keys:
            redis_client.delete(*keys)
        if cursor == 0:
            break
```

## Monitoring & Statistics

**Tracked Metrics:**
- Cache hit rate per cache type
- Average latency (cache hit vs miss)
- Total cache size (Redis memory usage)
- Document cache size (S3 storage used)
- Cost per query (with vs without cache)

**Endpoint**: `GET /stats`

**Response:**
```json
{
  "cache_statistics": {
    "rag": {
      "hits": 456,
      "misses": 234,
      "hit_rate": 0.66,
      "avg_latency_ms": 7
    },
    "embeddings": {
      "hits": 789,
      "misses": 321,
      "hit_rate": 0.71,
      "avg_latency_ms": 6
    },
    "sql_generation": {
      "hits": 234,
      "misses": 156,
      "hit_rate": 0.60,
      "avg_latency_ms": 8
    },
    "sql_results": {
      "hits": 123,
      "misses": 187,
      "hit_rate": 0.40,
      "avg_latency_ms": 9
    }
  },
  "document_cache": {
    "total_documents": 45,
    "total_size_mb": 123.4,
    "storage_backend": "s3"
  }
}
```

## Important Notes

- **Redis Eviction**: Set `maxmemory-policy=allkeys-lru` to auto-evict old keys when memory full
- **S3 Lifecycle**: Configure lifecycle rules to delete old cached documents after 30-90 days
- **TTL Selection**: Balance freshness vs cost (shorter TTL = more API calls but fresher data)
- **Cache Warming**: Pre-populate cache for common queries during deployment
- **Monitoring**: Track hit rates and adjust TTLs/strategies accordingly

## Related Workflows
- [RAG Query Execution](04-rag-query-execution.md) - How RAG uses caching
- [SQL Query Execution](05-sql-query-execution.md) - How SQL uses caching
- [Document Upload Pipeline](03-document-upload-pipeline.md) - Document cache usage
- [Cache Invalidation](08-cache-invalidation.md) - Detailed invalidation strategies
