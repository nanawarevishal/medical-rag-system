# AWS Lambda Deployment Architecture - Mermaid Diagrams

**Project:** Multi-Source RAG + Text-to-SQL System
**Architecture:** AWS Lambda (x86-64) + Function URL + ECR + S3
**Last Updated:** 2026-01-25
**Deployment Method:** GitHub Actions CI/CD

---

## 1. CI/CD Deployment Flow

This diagram shows the automated deployment pipeline from code push to production.

```mermaid
graph TB
    subgraph "Developer Workflow"
        A[Developer] -->|git push| B[GitHub Repository]
    end

    subgraph "GitHub Actions CI/CD"
        B -->|Trigger on push to main| C[Checkout Code]
        C --> D[Set up QEMU]
        D --> E[Set up Docker Buildx]
        E --> F[Configure AWS Credentials]
        F --> G[Login to ECR]
        G --> H[Build Docker Image<br/>platform: linux/amd64]
        H --> I[Tag Image<br/>latest, amd64, commit-sha]
        I --> J[Push to ECR]
        J --> K[Update Lambda Function]
        K --> L[Configure Environment Variables]
        L --> M[Run Integration Tests]
        M --> N{Tests Pass?}
        N -->|Yes| O[✅ Deployment Success]
        N -->|No| P[❌ Rollback & Notify]
    end

    subgraph "AWS Infrastructure"
        J --> ECR[(Amazon ECR<br/>Docker Registry)]
        K --> Lambda[AWS Lambda<br/>rag-text-to-sql-serverless<br/>x86-64, 3GB RAM, 15min timeout]
        Lambda --> FunctionURL[Lambda Function URL<br/>Public HTTPS Endpoint]
    end

    style O fill:#90EE90
    style P fill:#FFB6C1
    style Lambda fill:#FF9900
    style ECR fill:#FF9900
    style FunctionURL fill:#00B0FF
```

---

## 2. Runtime Architecture & Request Flow

This diagram shows how requests flow through the system at runtime.

```mermaid
graph TB
    subgraph "Client Layer"
        Client[External Client<br/>curl, browser, mobile app]
    end

    subgraph "AWS Lambda + Function URL"
        Client -->|HTTPS Request<br/>15-min timeout| FunctionURL[Lambda Function URL<br/>https://...lambda-url.us-east-1.on.aws/]
        FunctionURL -->|Invoke| Lambda[AWS Lambda Function<br/>rag-text-to-sql-serverless]

        Lambda --> Handler[lambda_handler.py<br/>Mangum ASGI Adapter]
        Handler --> FastAPI[FastAPI Application<br/>app/main.py]

        FastAPI --> Router{Query Router<br/>Intelligent Routing}
        Router -->|Document Query| RAGService[RAG Service]
        Router -->|SQL Query| SQLService[SQL Service]
        Router -->|Hybrid Query| Both[RAG + SQL Combined]
    end

    subgraph "AWS Services"
        Lambda --> S3[(S3 Bucket<br/>Document Cache<br/>rag-cache-*)]
        Lambda --> CloudWatch[CloudWatch Logs<br/>Monitoring & Debugging]
    end

    subgraph "External Services"
        RAGService --> Pinecone[(Pinecone<br/>Vector Database<br/>rag-documents)]
        RAGService --> OpenAI1[OpenAI API<br/>Embeddings<br/>text-embedding-3-small]
        RAGService --> OpenAI2[OpenAI API<br/>LLM<br/>gpt-4o]

        SQLService --> Supabase[(Supabase PostgreSQL<br/>Session Pooler IPv4)]
        SQLService --> PineconeSQL[(Pinecone<br/>SQL Training<br/>vanna-sql-training)]
        SQLService --> OpenAI3[OpenAI API<br/>LLM for SQL Generation]

        RAGService --> Redis[(Upstash Redis<br/>Query Cache<br/>optional)]
        SQLService --> Redis
    end

    subgraph "Response Flow"
        FastAPI --> Response[JSON Response]
        Response --> FunctionURL
        FunctionURL --> Client
    end

    style Lambda fill:#FF9900
    style FunctionURL fill:#00B0FF
    style FastAPI fill:#009688
    style S3 fill:#569A31
    style Pinecone fill:#8B5CF6
    style Supabase fill:#3ECF8E
    style OpenAI1 fill:#10A37F
    style OpenAI2 fill:#10A37F
    style OpenAI3 fill:#10A37F
    style Redis fill:#DC382D
    style CloudWatch fill:#FF9900
```


---


## 3. Cost Breakdown (Monthly Estimate)

```mermaid
pie title AWS Monthly Cost ($26.43 for 100K requests)
    "Lambda Compute (x86-64)" : 20.83
    "Lambda Requests" : 2.00
    "ECR Storage" : 0.40
    "CloudWatch Logs" : 2.50
    "S3 Storage" : 0.25
    "Data Transfer" : 0.45
```

```mermaid
pie title External Services Monthly Cost ($100-130)
    "Pinecone (2 indexes)" : 70
    "OpenAI (Embeddings + LLM)" : 20
    "Supabase (Free/Pro)" : 15
    "Upstash Redis (optional)" : 10
    "OPIK (optional)" : 0
```

---



## 4. Key Architecture Benefits

| Feature | Value | Benefit |
|---------|-------|---------|
| **Timeout** | 15 minutes (900s) | ✅ Handles large PDF processing (138s observed) |
| **No API Gateway** | $0 extra cost | ✅ Saves $0.10 per 100K requests |
| **x86-64 Architecture** | AMD64 | ✅ Stable PyTorch/ONNX, Dockling support |
| **Function URL** | Direct HTTPS access | ✅ Simpler architecture, fewer services |
| **S3 Cache** | Hash-based caching | ✅ Fast repeated uploads (cache_hit: true) |
| **Auto-scaling** | Lambda auto-scales | ✅ Handles traffic spikes automatically |
| **Container Image** | Docker via ECR | ✅ Full control over dependencies |

---



---

## Technical Specifications

### Lambda Configuration
- **Function Name:** `rag-text-to-sql-serverless`
- **Architecture:** x86-64 (AMD64)
- **Memory:** 3008 MB (3 GB)
- **Timeout:** 900 seconds (15 minutes)
- **Ephemeral Storage:** 10 GB (/tmp)
- **Runtime:** Container (Python 3.12)
- **Handler:** `lambda_handler.handler`

### Function URL Configuration
- **Auth Type:** NONE (public access)
- **CORS:** Allow all origins, methods, headers
- **Invoke Mode:** BUFFERED (wait for full response)
- **Permissions Required (2026):**
  - `lambda:InvokeFunctionUrl`
  - `lambda:InvokeFunction`

### Docker Image
- **Base Image:** `public.ecr.aws/lambda/python:3.12`
- **Platform:** linux/amd64
- **Size:** ~3.6 GB (optimized with multi-stage build)
- **Builder:** Docker Buildx with UV package manager
- **Tags:** `latest`, `amd64`, `{commit-sha}`

### Environment Variables
- `OPENAI_API_KEY` - OpenAI API key
- `PINECONE_API_KEY` - Pinecone API key
- `PINECONE_ENVIRONMENT` - us-east-1-aws
- `PINECONE_INDEX_NAME` - rag-documents
- `DATABASE_URL` - Supabase Session Pooler (IPv4)
- `S3_CACHE_BUCKET` - rag-cache-{account-id}
- `USE_DOCKLING` - true (x86-64 benefit)
- `STORAGE_BACKEND` - s3

---

## Quick Reference

### Test Endpoints
```bash
# Health check
curl "https://YOUR-FUNCTION-URL.lambda-url.us-east-1.on.aws/health"

# Upload document
curl -X POST "https://YOUR-FUNCTION-URL.lambda-url.us-east-1.on.aws/upload" \
  -F "file=@document.pdf"

# Query documents
curl -X POST "https://YOUR-FUNCTION-URL.lambda-url.us-east-1.on.aws/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the return policy?"}'
```

### Monitoring
```bash
# View logs
aws logs tail /aws/lambda/rag-text-to-sql-serverless --follow

# Check function status
aws lambda get-function --function-name rag-text-to-sql-serverless

# Get Function URL
aws lambda get-function-url-config --function-name rag-text-to-sql-serverless
```

### Deployment
```bash
# Trigger deployment
git push origin main

# Monitor in GitHub Actions
# https://github.com/YOUR-USERNAME/YOUR-REPO/actions
```

---

**Document Created:** 2026-01-25
**Architecture:** AWS Lambda (x86-64) + Function URL + ECR + S3
**Total Monthly Cost:** ~$125-157 (100K requests/month)
**Key Advantage:** 15-minute timeout for large PDF processing
