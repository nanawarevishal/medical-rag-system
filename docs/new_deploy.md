# Complete AWS Deployment Guide for Fresh Account
## Multi-Source RAG + Text-to-SQL Systems

**Last Updated:** 2026-01-25
**Target Architecture:** AWS Lambda (x86-64/AMD64) + Lambda Function URL + ECR + S3
**Deployment Method:** GitHub Actions CI/CD
**Audience:** Intermediate AWS users familiar with CLI and console

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites & Requirements](#2-prerequisites--requirements)
3. [AWS IAM Setup](#3-aws-iam-setup)
4. [AWS Infrastructure Setup](#4-aws-infrastructure-setup)
5. [External Services Configuration](#5-external-services-configuration)
6. [GitHub Repository Setup](#6-github-repository-setup)
7. [Database Initialization](#7-database-initialization)
8. [Initial Deployment](#8-initial-deployment)
9. [Post-Deployment Validation](#9-post-deployment-validation)
10. [Troubleshooting Common Issues](#10-troubleshooting-common-issues)
11. [Monitoring & Maintenance](#11-monitoring--maintenance)
12. [Cost Summary](#12-cost-summary)
13. [Next Steps](#13-next-steps)

---

## ⚠️ Architecture Migration Notice (2026-01-25)

**This deployment guide has been updated to use x86-64 (AMD64) architecture instead of ARM64.**

**Migration Reason:**
- ARM64 Lambda had compatibility issues with PyTorch CPU detection
- ONNX Runtime stability problems on ARM64
- Dockling document parser (advanced PDF processing) was disabled on ARM64

**Benefits of x86-64:**
- ✅ Stable PyTorch and ONNX Runtime support
- ✅ Can enable Dockling for better document parsing quality
- ✅ Wider Python package ecosystem compatibility
- ✅ No CPU detection bugs or runtime crashes

**Trade-off:**
- ❌ ~20% higher Lambda costs (~$4-5/month additional for typical usage)
- ✅ Better stability and reliability for production deployments

**If you deployed with ARM64 before 2026-01-25:**
The GitHub Actions workflow will automatically rebuild and deploy using x86-64 on your next push to main. No manual intervention required.

---

## 1. Overview

### 1.1 What Will Be Deployed

This guide walks you through deploying a production-ready Multi-Source RAG (Retrieval-Augmented Generation) + Text-to-SQL system on AWS from scratch. The system combines document-based question answering with natural language database queries.

**Key Features:**
- 📄 **Document RAG:** Upload PDFs/DOCX/CSV/JSON, query them with natural language
- 🗃️ **Text-to-SQL:** Ask database questions in plain English, get SQL results
- 🧠 **Intelligent Routing:** Automatically determines whether to query documents or database
- 🔄 **Automatic Deployment:** Push to GitHub main branch → auto-deploy to AWS
- 🔧 **Production-Ready:** x86-64 Lambda (better compatibility, stable PyTorch/ONNX support)

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                         │
│                     (Push to main branch)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GitHub Actions CI/CD                        │
│  • Build Docker image (linux/amd64)                             │
│  • Push to Amazon ECR                                           │
│  • Update Lambda function                                       │
│  • Configure Function URL                                       │
│  • Run integration tests                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Amazon ECR (Registry)                       │
│                    Docker Image Storage                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AWS Lambda (Serverless)                        │
│  • x86-64 architecture (AMD64)                                  │
│  • 3008 MB memory                                               │
│  • 900 second timeout (15 minutes)                              │
│  • 10 GB ephemeral storage                                      │
│  • FastAPI + Mangum adapter                                     │
│  • Function URL: Public HTTPS endpoint                          │
│  • 15-minute timeout (vs API Gateway's 30-second limit)         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTPS Function URL
                             │ (https://...lambda-url.us-east-1.on.aws/)
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
         ┌──────────────────┐  ┌────────────────┐
         │   S3 Bucket      │  │ External APIs  │
         │ Document Cache   │  │ • OpenAI       │
         │                  │  │ • Pinecone     │
         └──────────────────┘  │ • Supabase     │
                               │ • Upstash      │
                               │ • OPIK         │
                               └────────────────┘
```

### 1.3 Prerequisites Checklist

Before starting, ensure you have:

**Local Environment:**
- [ ] AWS CLI v2.x installed and configured
- [ ] Git installed
- [ ] Docker installed (for local testing)
- [ ] GitHub account with repository access

**AWS Account:**
- [ ] Active AWS account with billing enabled
- [ ] Admin access or permissions for ECR, Lambda, IAM, S3, CloudWatch
- [ ] Credit card on file (AWS requires it even for free tier)

**External Service Accounts:**
- [ ] OpenAI API key (https://platform.openai.com/api-keys)
- [ ] Pinecone account and API key (https://www.pinecone.io/)
- [ ] Supabase account with PostgreSQL database (https://supabase.com/)
- [ ] (Optional) Upstash Redis for query caching (https://upstash.com/)
- [ ] (Optional) OPIK monitoring account (https://www.comet.com/opik)

**Estimated Time:** 60-90 minutes for complete setup

---

## 2. Prerequisites & Requirements

### 2.1 Install AWS CLI

The AWS Command Line Interface is required for managing AWS resources.

**macOS:**
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
rm AWSCLIV2.pkg
```

**Linux (x86_64):**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip
```

**Linux (ARM64):**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip
```

**Windows:**
- Download: https://awscli.amazonaws.com/AWSCLIV2.msi
- Run the installer
- Restart your terminal

**Verify Installation:**
```bash
aws --version
# Expected output: aws-cli/2.x.x Python/3.x.x ...
```

### 2.2 Install Docker (Optional but Recommended)

Docker is needed for local testing before deployment.

**macOS:**
- Download Docker Desktop: https://www.docker.com/products/docker-desktop/
- Install and start Docker Desktop

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker  # Activate group without logout
rm get-docker.sh
```

**Windows:**
- Download Docker Desktop: https://www.docker.com/products/docker-desktop/
- Install WSL2 if prompted
- Start Docker Desktop

**Verify Installation:**
```bash
docker --version
# Expected output: Docker version 20.x.x or higher
```

### 2.3 Configure AWS CLI

You'll need AWS credentials to interact with your account.

**Step 1: Create IAM User (if you don't have one)**

1. Log into AWS Console: https://console.aws.amazon.com/
2. Navigate to **IAM** → **Users** → **Create user**
3. User name: `deployment-user` (or your preferred name)
4. Select **Programmatic access** (Access key - Programmatic access)
5. Click **Next**

**Step 2: Attach Permissions**

For simplicity during setup, attach `AdministratorAccess` policy:
- Search for `AdministratorAccess`
- Check the box
- Click **Next** → **Create user**

⚠️ **Security Note:** This user will be used for both AWS CLI access and GitHub Actions deployment. We'll add the necessary deployment permissions in Section 3.2.

**Step 3: Generate Access Keys**

1. Click on the created user
2. Go to **Security credentials** tab
3. Click **Create access key**
4. Select "Command Line Interface (CLI)"
5. Check "I understand the above recommendation"
6. Click **Next** → **Create access key**
7. **Download the CSV file** or copy both:
   - Access Key ID (starts with `AKIA...`)
   - Secret Access Key (long random string)

⚠️ **CRITICAL:** Save these credentials securely. You won't be able to see the secret key again!

**Step 4: Configure AWS CLI**

Run the configuration command:

```bash
aws configure
```

You'll be prompted for:

```
AWS Access Key ID [None]: AKIA...YOUR_KEY_HERE
AWS Secret Access Key [None]: YOUR_SECRET_KEY_HERE
Default region name [None]: us-east-1
Default output format [None]: json
```

**Explanation:**
- **Access Key ID:** Your AWS identity
- **Secret Access Key:** Your AWS password (keep this secret!)
- **Region:** `us-east-1` (US East - Virginia) - cheapest, most features available
- **Output format:** `json` - easiest to read and parse

**Step 5: Verify Access**

Test that AWS CLI works:

```bash
aws sts get-caller-identity
```

**Expected output:**
```json
{
    "UserId": "AIDACKCEVSQ6C2EXAMPLE",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/deployment-user"
}
```

**Save your AWS Account ID** - you'll need it later:

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Your AWS Account ID: $AWS_ACCOUNT_ID"
```

---

## 3. AWS IAM Setup

### 3.1 Create Lambda Execution Role

Lambda needs an IAM role to access AWS services (CloudWatch logs, S3, etc.).

**Step 1: Create Trust Policy**

This policy allows Lambda to assume the role:

```bash
cat > /tmp/lambda-trust-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
```

**Step 2: Create the IAM Role**

```bash
aws iam create-role \
  --role-name rag-lambda-execution-role \
  --assume-role-policy-document file:///tmp/lambda-trust-policy.json
```

**Expected output:**
```json
{
    "Role": {
        "Path": "/",
        "RoleName": "rag-lambda-execution-role",
        "RoleId": "AROAEXAMPLEROLEID",
        "Arn": "arn:aws:iam::123456789012:role/rag-lambda-execution-role",
        "CreateDate": "2026-01-22T10:00:00Z"
    }
}
```

**If you see "EntityAlreadyExists":** The role already exists - that's fine, continue!

**Step 3: Attach Basic Lambda Execution Policy**

This gives Lambda permission to write logs to CloudWatch:

```bash
aws iam attach-role-policy \
  --role-name rag-lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

**No output = success!**

**Step 4: Add S3 Permissions**

Lambda needs to read/write to S3 for document caching:

```bash
cat > /tmp/lambda-s3-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::rag-cache-*",
        "arn:aws:s3:::rag-cache-*/*"
      ]
    }
  ]
}
EOF
```

Create and attach the policy:

```bash
aws iam create-policy \
  --policy-name rag-lambda-s3-access \
  --policy-document file:///tmp/lambda-s3-policy.json

aws iam attach-role-policy \
  --role-name rag-lambda-execution-role \
  --policy-arn arn:aws:iam::${AWS_ACCOUNT_ID}:policy/rag-lambda-s3-access
```

### 3.2 Add Deployment Permissions to User

Add deployment permissions to the user created in Section 2.3. This allows the same user to deploy via GitHub Actions while following the principle of least privilege.

**Step 1: Create Deployment Policy**

```bash
cat > /tmp/github-actions-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRPermissions",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:DescribeImages"
      ],
      "Resource": "*"
    },
    {
      "Sid": "LambdaUpdatePermissions",
      "Effect": "Allow",
      "Action": [
        "lambda:UpdateFunctionCode",
        "lambda:UpdateFunctionConfiguration",
        "lambda:GetFunction",
        "lambda:GetFunctionConfiguration"
      ],
      "Resource": "arn:aws:lambda:us-east-1:${AWS_ACCOUNT_ID}:function:rag-text-to-sql-serverless"
    },
    {
      "Sid": "S3TestPermissions",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::rag-cache-*",
        "arn:aws:s3:::rag-cache-*/*"
      ]
    },
    {
      "Sid": "CloudWatchLogsRead",
      "Effect": "Allow",
      "Action": [
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams",
        "logs:GetLogEvents",
        "logs:FilterLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
EOF
```

**Step 2: Create and Attach Policy**

```bash
aws iam create-policy \
  --policy-name rag-deployment-policy \
  --policy-document file:///tmp/github-actions-policy.json

aws iam attach-user-policy \
  --user-name deployment-user \
  --policy-arn arn:aws:iam::${AWS_ACCOUNT_ID}:policy/rag-deployment-policy
```

**Note:** If you used a different user name in Section 2.3, replace `deployment-user` with your actual IAM user name.

---

## 4. AWS Infrastructure Setup

Now we'll create all AWS resources needed for the application.

### 4.1 Create S3 Bucket for Document Cache

Lambda uses S3 to store processed documents and embeddings persistently.

**Step 1: Generate Unique Bucket Name**

S3 bucket names must be globally unique. Use your account ID as a suffix:

```bash
export S3_BUCKET_NAME="rag-cache-${AWS_ACCOUNT_ID}"
echo "Your S3 bucket will be: $S3_BUCKET_NAME"
```

**Step 2: Create the Bucket**

```bash
aws s3 mb s3://${S3_BUCKET_NAME} --region us-east-1
```

**Expected output:**
```
make_bucket: rag-cache-123456789012
```

**Step 3: Enable Versioning (Optional but Recommended)**

Versioning allows recovery from accidental deletions:

```bash
aws s3api put-bucket-versioning \
  --bucket ${S3_BUCKET_NAME} \
  --versioning-configuration Status=Enabled
```

**Step 4: Configure Lifecycle Policy (Optional)**

Automatically delete old cached documents after 30 days to save costs:

```bash
cat > /tmp/s3-lifecycle.json <<'EOF'
{
  "Rules": [
    {
      "ID": "DeleteOldCache",
      "Status": "Enabled",
      "Filter": {},
      "Expiration": {
        "Days": 30
      },
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 7
      }
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket ${S3_BUCKET_NAME} \
  --lifecycle-configuration file:///tmp/s3-lifecycle.json
```

**Step 5: Verify Bucket**

```bash
aws s3 ls | grep rag-cache
```

**Expected output:**
```
2026-01-22 10:00:00 rag-cache-123456789012
```

### 4.2 Create ECR Repository

Amazon Elastic Container Registry (ECR) stores your Docker images.

```bash
aws ecr create-repository \
  --repository-name rag-text-to-sql \
  --region us-east-1 \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256
```

**Expected output:**
```json
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:123456789012:repository/rag-text-to-sql",
        "registryId": "123456789012",
        "repositoryName": "rag-text-to-sql",
        "repositoryUri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/rag-text-to-sql",
        "createdAt": "2026-01-22T10:00:00+00:00"
    }
}
```

**Save the `repositoryUri`** - you'll need it:

```bash
export ECR_URI=$(aws ecr describe-repositories \
  --repository-names rag-text-to-sql \
  --query 'repositories[0].repositoryUri' \
  --output text)

echo "ECR Repository URI: $ECR_URI"
```

**If you see "RepositoryAlreadyExistsException":** That's fine, continue!

### 4.3 Build and Push Initial Docker Image

Before creating the Lambda function, we need an initial Docker image in ECR.

**Step 1: Clone the Repository**

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/multidata-rag-project.git
cd multidata-rag-project
```

**Step 2: Authenticate Docker to ECR**

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin ${ECR_URI}
```

**Expected output:**
```
Login Succeeded
```

**Step 3: Build Docker Image for x86-64**

```bash
docker build \
  --platform linux/amd64 \
  -f Dockerfile.lambda \
  -t ${ECR_URI}:latest \
  .
```

**This will take 5-10 minutes** depending on your internet speed and CPU.

**Why x86-64?**
- Better compatibility with PyTorch and ONNX Runtime (no CPU detection issues)
- Stable support for Dockling document parser
- Mature ecosystem with more optimized Python packages
- Resolves ARM64-specific dependency issues

**Note:** x86-64 Lambda costs ~20% more than ARM64, but provides better stability and compatibility.

**Step 4: Tag Image**

```bash
docker tag ${ECR_URI}:latest ${ECR_URI}:amd64
docker tag ${ECR_URI}:latest ${ECR_URI}:$(git rev-parse --short HEAD)
```

**Step 5: Push to ECR**

```bash
docker push ${ECR_URI}:latest
docker push ${ECR_URI}:amd64
```

**This will take 3-5 minutes** to upload ~3.6 GB image.

**Step 6: Verify Push**

```bash
aws ecr describe-images \
  --repository-name rag-text-to-sql \
  --query 'imageDetails[*].imageTags' \
  --output table
```

**Expected output:**
```
-----------------------
|  DescribeImages     |
+---------------------+
|  latest            |
|  amd64             |
+---------------------+
```

### 4.4 Create Lambda Function

Now we can create the Lambda function using the Docker image.

**Step 1: Create the Function**

```bash
aws lambda create-function \
  --function-name rag-text-to-sql-serverless \
  --package-type Image \
  --code ImageUri=${ECR_URI}:amd64 \
  --role arn:aws:iam::${AWS_ACCOUNT_ID}:role/rag-lambda-execution-role \
  --architectures x86_64 \
  --timeout 900 \
  --memory-size 3008 \
  --ephemeral-storage Size=10240 \
  --description "Multi-Source RAG + Text-to-SQL API" \
  --region us-east-1
```

**Configuration Explained:**
- `--function-name`: Name of your Lambda function
- `--package-type Image`: We're using Docker container (not ZIP)
- `--code ImageUri`: Points to the x86-64 Docker image in ECR
- `--role`: The IAM role we created earlier
- `--architectures x86_64`: x86-64 for better compatibility (PyTorch, ONNX)
- `--timeout 900`: Max execution time (15 minutes - for large PDF processing)
- `--memory-size 3008`: 3 GB RAM (needed for document processing)
- `--ephemeral-storage`: 10 GB temporary disk space in `/tmp`

**Expected output:**
```json
{
    "FunctionName": "rag-text-to-sql",
    "FunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:rag-text-to-sql-serverless",
    "Role": "arn:aws:iam::123456789012:role/rag-lambda-execution-role",
    "CodeSize": 0,
    "Runtime": null,
    "Handler": null,
    "MemorySize": 3008,
    "Timeout": 300,
    "State": "Pending",
    "Architectures": ["x86_64"]
}
```

**Step 2: Wait for Function to be Active**

```bash
echo "Waiting for Lambda function to be ready..."
aws lambda wait function-active --function-name rag-text-to-sql
echo "✅ Lambda function is active!"
```

This may take 30-60 seconds.

**Step 3: Verify Function**

```bash
aws lambda get-function \
  --function-name rag-text-to-sql-serverless \
  --query 'Configuration.{State:State,Memory:MemorySize,Timeout:Timeout,Arch:Architectures[0]}' \
  --output table
```

**Expected output:**
```
----------------------------------------------
|            GetFunction                      |
+--------+--------+-----------+-------------+
|  Arch  | Memory |   State   |   Timeout   |
+--------+--------+-----------+-------------+
| x86_64 |  3008  |  Active   |     300     |
+--------+--------+-----------+-------------+
```

### 4.5 Create Lambda Function URL

Lambda Function URLs provide a direct public HTTPS endpoint to your function without needing API Gateway.

**Why Function URL instead of API Gateway?**
- ✅ **15-minute timeout** (vs API Gateway's 30-second hard limit)
- ✅ **No extra cost** (API Gateway charges $1 per million requests)
- ✅ **Simpler architecture** (one less service to manage)
- ✅ **Perfect for long-running PDF processing** (138+ seconds)
- ❌ No built-in throttling/rate limiting (use AWS WAF if needed)
- ❌ No API key authentication (implement in code or use AWS IAM)

**Step 1: Create Function URL**

```bash
FUNCTION_URL=$(aws lambda create-function-url-config \
  --function-name rag-text-to-sql-serverless \
  --auth-type NONE \
  --cors "AllowOrigins=*,AllowMethods=*,AllowHeaders=*" \
  --invoke-mode BUFFERED \
  --query 'FunctionUrl' \
  --output text)

echo "Lambda Function URL created: $FUNCTION_URL"
```

**Configuration Explained:**
- `--auth-type NONE`: Public access (no authentication required)
- `--cors`: Allow cross-origin requests from any domain
- `--invoke-mode BUFFERED`: Wait for response (vs RESPONSE_STREAM for streaming)

**Step 2: Grant Public Access Permission (2026 Authorization Model)**

⚠️ **CRITICAL:** AWS changed Lambda Function URL authorization in 2026 to require **TWO** permissions:

```bash
# Permission 1: InvokeFunctionUrl (for Function URL access)
aws lambda add-permission \
  --function-name rag-text-to-sql-serverless \
  --statement-id FunctionURLInvokeFunctionUrl \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE

# Permission 2: InvokeFunction (for public invocation - NEW in 2026)
aws lambda add-permission \
  --function-name rag-text-to-sql-serverless \
  --statement-id FunctionURLInvokeFunction \
  --action lambda:InvokeFunction \
  --principal "*"
```

**Why two permissions?**
AWS enhanced security in 2026 to require both `lambda:InvokeFunctionUrl` AND `lambda:InvokeFunction` permissions. Without both, you'll get "403 Forbidden" errors.

**Step 3: Save Function URL**

```bash
export API_URL="${FUNCTION_URL}"
echo ""
echo "========================================="
echo "🎉 Your Lambda Function URL is Live!"
echo "========================================="
echo "Function URL: $API_URL"
echo "Timeout: 15 minutes (900 seconds)"
echo "Cost: $0 extra (no API Gateway charges)"
echo ""
echo "Test with:"
echo "  curl ${API_URL}health"
echo "========================================="
```

**Save this URL!** You'll need it for GitHub configuration and testing.

**Example URL:** `https://abc123xyz.lambda-url.us-east-1.on.aws/`

**Step 4: Verify Function URL Configuration**

```bash
aws lambda get-function-url-config \
  --function-name rag-text-to-sql-serverless
```

**Expected output:**
```json
{
    "FunctionUrl": "https://abc123xyz.lambda-url.us-east-1.on.aws/",
    "FunctionArn": "arn:aws:lambda:us-east-1:123456789012:function/rag-text-to-sql-serverless-serverless",
    "AuthType": "NONE",
    "Cors": {
        "AllowOrigins": ["*"],
        "AllowMethods": ["*"],
        "AllowHeaders": ["*"]
    },
    "CreationTime": "2026-01-25T10:00:00Z",
    "InvokeMode": "BUFFERED"
}
```

---

## 5. External Services Configuration

The application requires several external services for full functionality.

### 5.1 OpenAI Setup

OpenAI provides embeddings (text-embedding-3-small) and LLM (GPT-4) for RAG and SQL generation.

**Step 1: Create Account**
1. Go to https://platform.openai.com/
2. Sign up or log in
3. Add payment method (required for API access)

**Step 2: Get API Key**
1. Navigate to https://platform.openai.com/api-keys
2. Click **Create new secret key**
3. Name: `rag-text-to-sql-production`
4. **Copy the key** (starts with `sk-...`)
5. Save it securely - you can't view it again!

**Step 3: Set Usage Limits (Recommended)**
1. Go to https://platform.openai.com/account/limits
2. Set monthly budget limit (e.g., $50)
3. Enable email notifications for usage alerts

**Expected Monthly Cost:** $10-30 depending on usage
- text-embedding-3-small: $0.0001 per 1K tokens
- GPT-4o: $5 per 1M input tokens, $15 per 1M output tokens

### 5.2 Pinecone Setup

Pinecone stores vector embeddings for document search and SQL training.

**Step 1: Create Account**
1. Go to https://www.pinecone.io/
2. Sign up for free account
3. Verify email

**Step 2: Create First Index (RAG Documents)**
1. Log into Pinecone console
2. Click **Create Index**
3. Configuration:
   - **Index name:** `rag-documents`
   - **Dimensions:** `1536` (must match OpenAI text-embedding-3-small)
   - **Metric:** `cosine`
   - **Region:** `us-east-1` (same as your AWS region)
   - **Plan:** Starter (free) or Serverless
4. Click **Create Index**
5. Wait 30-60 seconds for index to be ready

**Step 3: Create Second Index (SQL Training)**
1. Click **Create Index** again
2. Configuration:
   - **Index name:** `vanna-sql-training`
   - **Dimensions:** `1536`
   - **Metric:** `cosine`
   - **Region:** `us-east-1`
   - **Plan:** Starter (free) or Serverless
3. Click **Create Index**

**Step 4: Get API Key**
1. Click on your project name (top left)
2. Go to **API Keys** tab
3. Copy the **API Key**
4. Copy the **Environment** (e.g., `us-east-1-aws`)

**Expected Monthly Cost:** $70-100 for Starter plan, or pay-as-you-go for Serverless

### 5.3 Supabase/PostgreSQL Setup

Supabase provides a managed PostgreSQL database for Text-to-SQL queries.

**Step 1: Create Project**
1. Go to https://supabase.com/
2. Sign up or log in
3. Click **New Project**
4. Configuration:
   - **Name:** `rag-text-to-sql`
   - **Database Password:** (generate strong password and save it!)
   - **Region:** US East (closest to `us-east-1`)
   - **Plan:** Free tier (sufficient for testing)
5. Click **Create new project**
6. Wait 2-3 minutes for provisioning

**Step 2: Get Session Pooler Connection String**

⚠️ **CRITICAL:** AWS Lambda requires IPv4 connections. You MUST use the **Session Pooler**, not Direct Connection.

1. Go to **Project Settings** → **Database**
2. Scroll to **Connection string**
3. Select **Session Pooler** (NOT "Direct connection")
4. Copy the connection string:
   ```
   postgres://postgres.[PROJECT-REF]:[YOUR-PASSWORD]@aws-1-us-east-1.pooler.supabase.com:5432/postgres
   ```
5. Replace `[YOUR-PASSWORD]` with your actual database password
6. Save this connection string securely

**Why Session Pooler?**
- Lambda doesn't support IPv6 outbound connections
- Session Pooler proxies through IPv4
- Direct connection will fail with connection timeout errors

**Step 3: Get Database Schema SQL**

You'll initialize the database in Section 7.

**Expected Monthly Cost:** Free tier (500 MB database) or $25/month for Pro plan

### 5.4 Upstash Redis Setup (Optional)

Upstash Redis provides query-level caching for faster responses.

**⚠️ Note:** This is **optional**. The application works without Redis, just slightly slower on repeated queries.

**Step 1: Create Account**
1. Go to https://console.upstash.com/
2. Sign up with GitHub/Google or email

**Step 2: Create Redis Database**
1. Click **Create Database**
2. Configuration:
   - **Name:** `rag-query-cache`
   - **Type:** Regional (cheaper)
   - **Region:** US-East-1 (closest to Lambda)
   - **TLS:** Enabled (recommended)
   - **Eviction:** allkeys-lru (automatically remove old keys)
3. Click **Create**

**Step 3: Get Credentials**
1. Click on your database
2. Scroll to **REST API** section
3. Copy:
   - **UPSTASH_REDIS_REST_URL**
   - **UPSTASH_REDIS_REST_TOKEN**

**Expected Monthly Cost:** Free tier (10K requests/day) or $0.20 per 100K requests

**Skip this section if:** You want to save costs and don't mind slightly slower repeated queries.

### 5.5 OPIK Monitoring Setup (Optional)

OPIK provides LLM observability and evaluation metrics.

**⚠️ Note:** This is **optional**. Skip this if you don't need advanced monitoring.

**Step 1: Create Account**
1. Go to https://www.comet.com/opik
2. Sign up for free account

**Step 2: Create Project**
1. Log into OPIK
2. Create new project: `Multi-Source-RAG`

**Step 3: Get API Key**
1. Go to **Settings** → **API Keys**
2. Copy your API key

**Expected Monthly Cost:** Free tier available

---

## 6. GitHub Repository Setup

Configure GitHub to automatically deploy on every push to main.

### 6.1 Fork or Clone Repository

**Option A: Fork (if using original repo)**
1. Go to https://github.com/ORIGINAL_REPO/multidata-rag-project
2. Click **Fork** (top right)
3. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/multidata-rag-project.git
   cd multidata-rag-project
   ```

**Option B: Use Existing Repository**
```bash
cd /path/to/multidata-rag-project
```

### 6.2 Configure GitHub Secrets

GitHub Actions needs AWS credentials and API keys stored as encrypted secrets.

**Step 1: Navigate to Repository Settings**
1. Go to your GitHub repository
2. Click **Settings** (top navigation)
3. Click **Secrets and variables** → **Actions** (left sidebar)

**Step 2: Add Required Secrets**

Click **New repository secret** for each of these:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `AWS_ACCESS_KEY_ID` | `AKIA...` | IAM user access key from Section 2.3 |
| `AWS_SECRET_ACCESS_KEY` | `wJalr...` | IAM user secret key from Section 2.3 |
| `AWS_ACCOUNT_ID` | `123456789012` | Your 12-digit AWS account ID |
| `PINECONE_API_KEY` | `abc123...` | From Section 5.2 |
| `PINECONE_ENVIRONMENT` | `us-east-1-aws` | From Section 5.2 |
| `PINECONE_INDEX_NAME` | `rag-documents` | From Section 5.2 |
| `OPENAI_API_KEY` | `sk-proj...` | From Section 5.1 |
| `DATABASE_URL` | `postgres://...` | Session Pooler URL from Section 5.3 |
| `S3_CACHE_BUCKET` | `rag-cache-123456789012` | From Section 4.1 |
| `LAMBDA_FUNCTION_URL` | `https://...lambda-url...` | From Section 4.5 |

**Optional Secrets** (only if you configured these services):

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `UPSTASH_REDIS_URL` | `https://...upstash.io` | From Section 5.4 (optional) |
| `UPSTASH_REDIS_TOKEN` | `...` | From Section 5.4 (optional) |
| `OPIK_API_KEY` | `...` | From Section 5.5 (optional) |

**Step 3: Verify Secrets**

All required secrets should now be listed on the Actions secrets page.

### 6.3 Verify Workflow Configuration

Check that the deployment workflow is configured correctly:

```bash
cat .github/workflows/deploy.yml | head -20
```

You should see:
```yaml
name: Deploy to AWS Lambda

on:
  push:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: rag-text-to-sql
  LAMBDA_FUNCTION: rag-text-to-sql
```

If this file doesn't exist or is incorrect, the deployment won't work!

---

## 7. Database Initialization

Initialize your Supabase PostgreSQL database with the e-commerce schema.

### 7.1 Connect to Database

**Option A: Using Supabase SQL Editor (Easiest)**
1. Go to Supabase dashboard
2. Click on your project
3. Navigate to **SQL Editor** (left sidebar)
4. Click **New query**

**Option B: Using psql CLI**
```bash
psql "postgres://postgres.[PROJECT-REF]:[PASSWORD]@aws-1-us-east-1.pooler.supabase.com:5432/postgres"
```

### 7.2 Run Schema SQL

Copy and paste the entire contents of `data/sql/schema.sql`:

```bash
cat data/sql/schema.sql
```

**Or run directly with psql:**
```bash
psql "postgres://..." < data/sql/schema.sql
```

This creates three tables:
- `customers` - Customer information
- `products` - Product catalog
- `orders` - Order history

### 7.3 Generate Sample Data (Optional but Recommended)

Generate sample data for testing SQL queries:

```bash
# Install dependencies if not already installed
pip install faker psycopg2-binary

# Generate sample data
python data/generate_sample_data.py
```

This will:
- Create 100 customers
- Create 200 products
- Create 50 orders with line items

**Verify Data:**

```sql
SELECT COUNT(*) FROM customers;
SELECT COUNT(*) FROM products;
SELECT COUNT(*) FROM orders;
```

Expected output:
```
count
-------
  100
(1 row)

count
-------
  200
(1 row)

count
-------
   50
(1 row)
```

---

## 8. Initial Deployment

Now that everything is configured, let's deploy!

### 8.1 Set Lambda Environment Variables

Configure Lambda with all API keys and settings:

```bash
aws lambda update-function-configuration \
  --function-name rag-text-to-sql-serverless \
  --environment "Variables={
    ENVIRONMENT=production,
    OPENAI_API_KEY=${OPENAI_API_KEY},
    PINECONE_API_KEY=${PINECONE_API_KEY},
    PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT},
    PINECONE_INDEX_NAME=${PINECONE_INDEX_NAME},
    DATABASE_URL=${DATABASE_URL},
    S3_CACHE_BUCKET=${S3_BUCKET_NAME},
    STORAGE_BACKEND=s3,
    HOME=/tmp,
    TMPDIR=/tmp,
    TEMP=/tmp,
    TMP=/tmp,
    HF_HOME=/tmp/.cache/huggingface,
    TRANSFORMERS_CACHE=/tmp/.cache/huggingface,
    UNSTRUCTURED_CACHE_DIR=/tmp/.cache/unstructured,
    DOCLING_CACHE_DIR=/tmp/.cache/docling
  }"
```

**⚠️ Important:** Set these environment variables in your terminal first:
```bash
export OPENAI_API_KEY="sk-proj-..."
export PINECONE_API_KEY="..."
export PINECONE_ENVIRONMENT="us-east-1-aws"
export PINECONE_INDEX_NAME="rag-documents"
export DATABASE_URL="postgres://..."
```

**Optional variables** (add if you configured these services):
```bash
export UPSTASH_REDIS_URL="https://..."
export UPSTASH_REDIS_TOKEN="..."
export OPIK_API_KEY="..."
```

**Wait for Update:**
```bash
aws lambda wait function-updated --function-name rag-text-to-sql
echo "✅ Environment variables configured!"
```

**Optional: Enable Dockling Document Parser (x86-64 only)**

With x86-64 architecture, you can optionally enable Dockling for better document parsing quality:

```bash
aws lambda update-function-configuration \
  --function-name rag-text-to-sql-serverless \
  --environment "Variables={
    ENVIRONMENT=production,
    USE_DOCKLING=true,
    OPENAI_API_KEY=${OPENAI_API_KEY},
    PINECONE_API_KEY=${PINECONE_API_KEY},
    PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT},
    PINECONE_INDEX_NAME=${PINECONE_INDEX_NAME},
    DATABASE_URL=${DATABASE_URL},
    S3_CACHE_BUCKET=${S3_BUCKET_NAME},
    STORAGE_BACKEND=s3,
    HOME=/tmp,
    TMPDIR=/tmp,
    TEMP=/tmp,
    TMP=/tmp,
    HF_HOME=/tmp/.cache/huggingface,
    TRANSFORMERS_CACHE=/tmp/.cache/huggingface,
    UNSTRUCTURED_CACHE_DIR=/tmp/.cache/unstructured,
    DOCLING_CACHE_DIR=/tmp/.cache/docling
  }"

aws lambda wait function-updated --function-name rag-text-to-sql
```

**Note:** Dockling was disabled by default on ARM64 due to PyTorch/ONNX compatibility issues. With x86-64, you can safely enable it for improved PDF table extraction and layout analysis. Monitor CloudWatch logs after enabling to ensure no errors.

### 8.2 Test Lambda Function URL

Before setting up GitHub Actions, test the Lambda Function URL works:

```bash
echo "Testing Lambda Function URL..."
curl -s "${API_URL}health" | jq '.'
```

**Note:** Function URLs don't have a base path like API Gateway's `/prod`. URLs are:
- ✅ Correct: `https://....lambda-url.us-east-1.on.aws/health`
- ❌ Wrong: `https://....lambda-url.us-east-1.on.aws/prod/health`

**Expected response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-22T10:30:00",
  "environment": "production",
  "services": {
    "embedding_service": true,
    "vector_service": true,
    "rag_service": true,
    "sql_service": true
  }
}
```

**If services show as `false`:**
1. Check CloudWatch logs: `aws logs tail /aws/lambda/rag-text-to-sql-serverless --since 5m`
2. Verify environment variables: `aws lambda get-function-configuration --function-name rag-text-to-sql --query 'Environment.Variables'`
3. Check API keys are correct

### 8.3 Trigger GitHub Actions Deployment

Now let's trigger the first automated deployment:

**Step 1: Commit Your Changes** (if you made any local modifications)

```bash
git add .
git commit -m "Initial deployment configuration"
```

**Step 2: Push to Main Branch**

```bash
git push origin main
```

**Step 3: Monitor GitHub Actions**

1. Go to your GitHub repository
2. Click **Actions** tab
3. You should see "Deploy to AWS Lambda" workflow running
4. Click on it to see live progress

**Workflow Steps:**
1. ✓ Checkout code
2. ✓ Set up QEMU (for x86-64 cross-platform builds)
3. ✓ Set up Docker Buildx
4. ✓ Configure AWS credentials
5. ✓ Login to Amazon ECR
6. ✓ Build, tag, and push image to ECR (~5-8 minutes)
7. ✓ Update Lambda function
8. ✓ Configure Lambda environment
9. ✓ Test deployment
10. ✓ Test S3 storage backend

**Total deployment time:** ~10-15 minutes (subsequent deployments: ~5-8 minutes with caching)

### 8.4 Verify Deployment Success

When the workflow completes successfully:

**Check Workflow Status:**
- All steps should have green checkmarks ✅
- Last step should show "✅ DEPLOYMENT TEST PASSED!"

**Check Deployment Logs:**
```bash
# View the last deployment
aws logs tail /aws/lambda/rag-text-to-sql-serverless --since 10m
```

**Test the API:**
```bash
# Health check
curl "${API_URL}health" | jq '.'

# Info endpoint
curl "${API_URL}info" | jq '.'

# Stats endpoint
curl "${API_URL}stats" | jq '.'
```

---

## 9. Post-Deployment Validation

Thoroughly test all functionality to ensure everything works.

### 9.1 Test Health Endpoint

```bash
curl -X GET "${API_URL}health" | jq '.'
```

**Expected response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-22T10:45:00.123456",
  "environment": "production",
  "services": {
    "embedding_service": true,
    "vector_service": true,
    "rag_service": true,
    "sql_service": true
  }
}
```

**All services should be `true`.** If any are `false`, see Section 10 (Troubleshooting).

### 9.2 Test Document Upload

Create a test document:

```bash
cat > test-document.txt <<EOF
Company Policy: Remote Work

All employees are eligible for remote work up to 3 days per week.
Employees must maintain core working hours from 10 AM to 3 PM in their local timezone.
Weekly team meetings are mandatory on Mondays at 9 AM.

Vacation Policy:
- Full-time employees: 15 days per year
- Part-time employees: 7 days per year
- Unused vacation days do not roll over

Customer Support Hours:
Monday-Friday: 9 AM - 6 PM EST
Saturday: 10 AM - 2 PM EST
Sunday: Closed
EOF
```

Upload the document:

```bash
curl -X POST "${API_URL}upload" \
  -F "file=@test-document.txt" | jq '.'
```

**Expected response:**
```json
{
  "status": "success",
  "filename": "test-document.txt",
  "document_id": "abc123...",
  "chunks_created": 1,
  "storage_backend": "s3",
  "cache_hit": false,
  "processing_time_ms": 1234
}
```

**Verify:**
- ✅ `storage_backend` is `"s3"`
- ✅ `cache_hit` is `false` (first upload)
- ✅ `document_id` is returned

### 9.3 Test Document Query

Query the uploaded document:

```bash
curl -X POST "${API_URL}query/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the remote work hours?",
    "top_k": 3
  }' | jq '.'
```

**Expected response:**
```json
{
  "status": "success",
  "answer": "Employees working remotely must maintain core working hours from 10 AM to 3 PM in their local timezone.",
  "sources": [
    {
      "content": "All employees are eligible for remote work up to 3 days per week...",
      "metadata": {
        "filename": "test-document.txt",
        "chunk_index": 0
      },
      "similarity_score": 0.89
    }
  ],
  "query_type": "DOCUMENTS",
  "execution_time_ms": 2345
}
```

### 9.4 Test SQL Query

Query the database:

```bash
curl -X POST "${API_URL}query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many customers do we have?",
    "auto_approve_sql": true
  }' | jq '.'
```

**Expected response:**
```json
{
  "status": "success",
  "query_type": "SQL",
  "sql_generated": "SELECT COUNT(*) as total_customers FROM customers;",
  "results": [
    {
      "total_customers": 100
    }
  ],
  "execution_time_ms": 1567
}
```

### 9.5 Test Query Routing (Hybrid)

Test a query that needs both SQL and documents:

```bash
curl -X POST "${API_URL}query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me total sales and explain our vacation policy",
    "auto_approve_sql": true
  }' | jq '.'
```

**Expected response:**
```json
{
  "status": "success",
  "query_type": "HYBRID",
  "sql_results": {
    "sql": "SELECT SUM(total_amount) as total_sales FROM orders;",
    "results": [{"total_sales": 12345.67}]
  },
  "document_results": {
    "answer": "The vacation policy states that full-time employees get 15 days per year...",
    "sources": [...]
  },
  "execution_time_ms": 3456
}
```

### 9.6 Test S3 Cache

Re-upload the same document to verify S3 caching:

```bash
curl -X POST "${API_URL}upload" \
  -F "file=@test-document.txt" | jq '.'
```

**Expected response:**
```json
{
  "status": "success",
  "filename": "test-document.txt",
  "document_id": "abc123...",  # Same ID as before
  "chunks_created": 1,
  "storage_backend": "s3",
  "cache_hit": true,  # ← Should be true!
  "processing_time_ms": 123  # Much faster
}
```

**Verify:**
- ✅ `cache_hit` is now `true`
- ✅ `document_id` matches first upload
- ✅ Processing time is much lower (~100-300ms vs 1-3 seconds)

### 9.7 Verify S3 Storage

Check that documents are stored in S3:

```bash
aws s3 ls s3://${S3_BUCKET_NAME}/txt/ --recursive --human-readable
```

**Expected output:**
```
2026-01-22 10:50:00   1.2 KiB txt/abc123.../original.txt
2026-01-22 10:50:00   234 Bytes txt/abc123.../metadata.json
2026-01-22 10:50:00   567 Bytes txt/abc123.../chunks.json
```

### 9.8 Check CloudWatch Logs

View Lambda execution logs:

```bash
# Real-time logs (follow mode)
aws logs tail /aws/lambda/rag-text-to-sql-serverless --follow

# Logs from last 30 minutes
aws logs tail /aws/lambda/rag-text-to-sql-serverless --since 30m

# Search for errors
aws logs tail /aws/lambda/rag-text-to-sql-serverless --since 1h --filter-pattern "ERROR"
```

**Expected output:**
```
2026-01-22T10:45:00.000000+00:00 START RequestId: abc-123-def-456
2026-01-22T10:45:00.100000+00:00 INFO: Initializing services...
2026-01-22T10:45:02.000000+00:00 INFO: ✓ Document RAG services initialized!
2026-01-22T10:45:03.500000+00:00 INFO: ✓ Text-to-SQL service initialized!
2026-01-22T10:45:03.600000+00:00 INFO: ✓ Cache service initialized!
2026-01-22T10:45:03.750000+00:00 INFO: GET /health - 200 OK
2026-01-22T10:45:03.800000+00:00 END RequestId: abc-123-def-456
```

---

## 10. Troubleshooting Common Issues

### 10.1 GitHub Actions Deployment Fails

**Problem:** Workflow fails with "Unable to locate credentials"

**Solution:**
1. Verify GitHub secrets are set correctly:
   - Go to GitHub → Settings → Secrets → Actions
   - Check `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ACCOUNT_ID` exist
2. Verify IAM user has correct permissions (Section 3.2)
3. Test AWS CLI locally: `aws sts get-caller-identity`

---

### 10.2 Lambda Function Timeout

**Problem:** "Task timed out after 900.00 seconds"

**This should be rare** - 15-minute timeout is very generous for document processing.

**Solution:**
1. Check if document is excessively large (>100 MB)
2. Split large documents into smaller chunks
3. Check CloudWatch logs for actual bottleneck:
   ```bash
   aws logs tail /aws/lambda/rag-text-to-sql-serverless --since 30m --filter-pattern "REPORT"
   ```
4. Consider async processing for extremely large files

**Note:** Unlike API Gateway (30-second hard limit), Lambda Function URLs support up to 15 minutes - sufficient for most PDF processing tasks.

---

### 10.3 Lambda Function Returns 500 Internal Server Error

**Problem:** `{"message":"Internal server error"}` or blank response

**Root Causes:**
- Lambda function crashed
- Out of memory
- Missing environment variables
- Python exceptions

**Solution:**
1. Check Lambda logs for errors:
   ```bash
   aws logs tail /aws/lambda/rag-text-to-sql-serverless --since 10m --filter-pattern "ERROR"
   ```
2. Look for Python stack traces:
   ```bash
   aws logs tail /aws/lambda/rag-text-to-sql-serverless --since 10m | grep -A 10 "Traceback"
   ```
3. Verify environment variables:
   ```bash
   aws lambda get-function-configuration \
     --function-name rag-text-to-sql-serverless \
     --query 'Environment.Variables' | jq '.'
   ```
4. Increase memory if OOM errors:
   ```bash
   aws lambda update-function-configuration \
     --function-name rag-text-to-sql-serverless \
     --memory-size 4096  # 4 GB
   ```

---

### 10.4 Services Show as `false` in Health Check

**Problem:** Health endpoint returns `"embedding_service": false, ...`

**Root Cause:** Services failed to initialize (API keys invalid, connectivity issues)

**Solution:**

**Step 1: Check CloudWatch Logs**
```bash
aws logs tail /aws/lambda/rag-text-to-sql-serverless --since 5m | grep -i "error\|failed\|exception"
```

**Step 2: Verify Environment Variables**
```bash
aws lambda get-function-configuration \
  --function-name rag-text-to-sql-serverless \
  --query 'Environment.Variables' | jq '.'
```

Check that these are set and correct:
- `OPENAI_API_KEY` (starts with `sk-`)
- `PINECONE_API_KEY`
- `DATABASE_URL` (Session Pooler format)
- `S3_CACHE_BUCKET`

**Step 3: Test API Keys Manually**

Test OpenAI:
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | jq '.data[0]'
```

Test Pinecone:
```bash
# Check if indexes exist
curl -X GET "https://api.pinecone.io/indexes" \
  -H "Api-Key: $PINECONE_API_KEY" | jq '.'
```

Test Supabase:
```bash
psql "$DATABASE_URL" -c "SELECT 1;"
```

**Step 4: Redeploy with Correct Values**

If environment variables were wrong, update and wait:
```bash
aws lambda update-function-configuration \
  --function-name rag-text-to-sql-serverless \
  --environment Variables="{...}"  # Correct values

aws lambda wait function-updated --function-name rag-text-to-sql

# Test again
curl "${API_URL}health" | jq '.services'
```

---

### 10.5 Lambda Function URL Returns 403 Forbidden

**Problem:** `{"Message":"Forbidden. For troubleshooting Function URL authorization issues..."}`

**Root Cause:** Missing permissions due to AWS's 2026 authorization model change

**Solution:**

Lambda Function URLs require **BOTH** permissions since 2026:
1. `lambda:InvokeFunctionUrl` - for Function URL access
2. `lambda:InvokeFunction` - for public invocation

```bash
# Check current permissions
aws lambda get-policy \
  --function-name rag-text-to-sql-serverless \
  --query Policy --output text | jq '.Statement'

# Add missing permissions
aws lambda add-permission \
  --function-name rag-text-to-sql-serverless \
  --statement-id FunctionURLInvokeFunctionUrl \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE

aws lambda add-permission \
  --function-name rag-text-to-sql-serverless \
  --statement-id FunctionURLInvokeFunction \
  --action lambda:InvokeFunction \
  --principal "*"

# Wait 5-10 seconds for permission propagation
sleep 10

# Test again
curl "${API_URL}health"
```

**Why two permissions?**
AWS enhanced security in 2026 to require both permissions for public Function URL access. This prevents accidental public exposure.

---

### 10.6 Database Connection Errors

**Problem:** "could not connect to server", "connection timeout"

**Root Cause:** Using Direct Connection instead of Session Pooler (Lambda needs IPv4)

**Solution:**

**Step 1: Verify Connection String Format**

❌ **Wrong (Direct Connection - IPv6):**
```
postgresql://postgres.xxxxxxxxxxxx.supabase.co:5432/postgres
```

✅ **Correct (Session Pooler - IPv4):**
```
postgresql://postgres.xxxxxxxxxxxx:[PASSWORD]@aws-1-us-east-1.pooler.supabase.com:5432/postgres
```

**Step 2: Get Correct Connection String**
1. Go to Supabase dashboard
2. Project Settings → Database
3. **Connection string** → Select **"Session Pooler"** (not "Direct connection")
4. Copy the connection string

**Step 3: Update Lambda Environment Variable**
```bash
export DATABASE_URL="postgres://postgres.[PROJECT-REF]:[PASSWORD]@aws-1-us-east-1.pooler.supabase.com:5432/postgres"

aws lambda update-function-configuration \
  --function-name rag-text-to-sql-serverless \
  --environment Variables="{DATABASE_URL=${DATABASE_URL},...}"
```

**Step 4: Update GitHub Secret**
1. GitHub → Settings → Secrets → Actions
2. Edit `DATABASE_URL`
3. Paste Session Pooler connection string
4. Save

---

### 10.7 S3 Access Denied Errors

**Problem:** "AccessDenied" when uploading documents

**Solution:**

**Step 1: Verify Lambda Role Has S3 Permissions**
```bash
aws iam list-attached-role-policies \
  --role-name rag-lambda-execution-role
```

Should include `rag-lambda-s3-access` policy.

**Step 2: Verify Policy Allows Your Bucket**
```bash
aws iam get-policy-version \
  --policy-arn arn:aws:iam::${AWS_ACCOUNT_ID}:policy/rag-lambda-s3-access \
  --version-id v1 \
  --query 'PolicyVersion.Document' | jq '.'
```

Should include:
```json
{
  "Resource": [
    "arn:aws:s3:::rag-cache-*",
    "arn:aws:s3:::rag-cache-*/*"
  ]
}
```

**Step 3: Test S3 Access from Lambda**
```bash
aws lambda invoke \
  --function-name rag-text-to-sql-serverless \
  --payload '{"rawPath":"/health"}' \
  /tmp/response.json
```

Check logs for S3 errors.

---

### 10.8 Cold Start Too Slow (15+ seconds)

**Problem:** First request after idle period takes 15-40 seconds

**This is normal for Lambda!** Container initialization takes time.

**Solutions:**

**Option 1: Provisioned Concurrency (costs extra)**
```bash
aws lambda put-provisioned-concurrency-config \
  --function-name rag-text-to-sql-serverless \
  --provisioned-concurrent-executions 1 \
  --qualifier '$LATEST'
```

Cost: ~$10-15/month to keep 1 instance warm

**Option 2: Scheduled Ping (free)**

Create EventBridge rule to invoke Lambda every 5 minutes:
```bash
aws events put-rule \
  --name rag-lambda-warmer \
  --schedule-expression "rate(5 minutes)"

aws events put-targets \
  --rule rag-lambda-warmer \
  --targets Id=1,Arn=arn:aws:lambda:us-east-1:${AWS_ACCOUNT_ID}:function/rag-text-to-sql-serverless

aws lambda add-permission \
  --function-name rag-text-to-sql-serverless \
  --statement-id EventBridgeInvoke \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-1:${AWS_ACCOUNT_ID}:rule/rag-lambda-warmer
```

**Option 3: Accept Cold Starts**

For non-critical applications, 15-second cold starts are acceptable. Subsequent requests are fast (<2 seconds).

---

### 10.9 Lambda Out of Memory

**Problem:** "Runtime exited with error: signal: killed"

**Solution:**

Increase memory allocation:
```bash
aws lambda update-function-configuration \
  --function-name rag-text-to-sql-serverless \
  --memory-size 4096  # 4 GB (up to 10 GB available)
```

**Note:** More memory = more CPU and higher cost, but faster processing.

---

### 10.10 ECR Authentication Failures

**Problem:** GitHub Actions fails with "denied: Your authorization token has expired"

**Solution:**

This is usually temporary. GitHub Actions automatically re-authenticates. Just retry the workflow:
1. Go to Actions tab
2. Click failed workflow
3. Click "Re-run all jobs"

If problem persists:
1. Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in GitHub secrets
2. Check IAM user has `ecr:GetAuthorizationToken` permission

---

## 11. Monitoring & Maintenance

### 11.1 CloudWatch Metrics

**View Lambda Invocations:**
```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=rag-text-to-sql \
  --start-time $(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 3600 \
  --statistics Sum
```

**View Error Rate:**
```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Errors \
  --dimensions Name=FunctionName,Value=rag-text-to-sql \
  --start-time $(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 3600 \
  --statistics Sum
```

**View Duration:**
```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=rag-text-to-sql \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum
```

### 11.2 Set Up CloudWatch Alarms

**High Error Rate Alarm:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name rag-lambda-high-errors \
  --alarm-description "Alert when Lambda error rate exceeds 5%" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=FunctionName,Value=rag-text-to-sql
```

**Long Duration Alarm:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name rag-lambda-slow-response \
  --alarm-description "Alert when Lambda takes longer than 30 seconds" \
  --metric-name Duration \
  --namespace AWS/Lambda \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 30000 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=FunctionName,Value=rag-text-to-sql
```

### 11.3 Application Updates

**Automatic Updates (Recommended):**

Simply push to main:
```bash
git add .
git commit -m "Update feature X"
git push origin main
```

GitHub Actions automatically:
1. Builds new Docker image
2. Pushes to ECR
3. Updates Lambda function
4. Runs tests

**Manual Update (if needed):**
```bash
# Build new image
docker build --platform linux/amd64 -f Dockerfile.lambda -t ${ECR_URI}:manual .

# Push to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin ${ECR_URI}
docker push ${ECR_URI}:manual

# Update Lambda
aws lambda update-function-code \
  --function-name rag-text-to-sql-serverless \
  --image-uri ${ECR_URI}:manual \
  --architectures x86_64

# Wait for update
aws lambda wait function-updated --function-name rag-text-to-sql
```

### 11.4 Rollback Procedure

**View Previous Versions:**
```bash
aws ecr describe-images \
  --repository-name rag-text-to-sql \
  --query 'sort_by(imageDetails,& imagePushedAt)[*].[imageTags[0], imagePushedAt]' \
  --output table
```

**Rollback to Previous Version:**
```bash
# Get second-most-recent image tag
PREVIOUS_TAG=$(aws ecr describe-images \
  --repository-name rag-text-to-sql \
  --query 'sort_by(imageDetails,& imagePushedAt)[-2].imageTags[0]' \
  --output text)

echo "Rolling back to: $PREVIOUS_TAG"

# Update Lambda to previous version
aws lambda update-function-code \
  --function-name rag-text-to-sql-serverless \
  --image-uri ${ECR_URI}:${PREVIOUS_TAG} \
  --architectures x86_64

# Wait and verify
aws lambda wait function-updated --function-name rag-text-to-sql
curl ${API_URL}/health
```

### 11.5 Cost Monitoring

**View Current Month Costs:**
```bash
aws ce get-cost-and-usage \
  --time-period Start=$(date -d 'first day of this month' +%Y-%m-01),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=SERVICE
```

**Set Billing Alarm:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name billing-alert-50 \
  --alarm-description "Alert when monthly bill exceeds $50" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 21600 \
  --evaluation-periods 1 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=Currency,Value=USD
```

---

## 12. Cost Summary

### 12.1 AWS Services (Monthly Estimate)

Based on **100,000 requests/month**, **30-second average duration**, **3 GB RAM**, **x86-64 architecture**:

| Service | Configuration | Usage | Monthly Cost |
|---------|--------------|-------|--------------|
| **Lambda (x86-64)** | 3 GB RAM, 100K invocations, 30s avg | 833,333 GB-seconds | **$20.83** |
| **Lambda Requests** | 100K requests | 100K requests | **$2.00** |
| **ECR Storage** | ~4 GB (3 image versions) | 4 GB storage | **$0.40** |
| **CloudWatch Logs** | 5 GB logs, 7-day retention | 5 GB ingestion + storage | **$2.50** |
| **S3 Storage** | 10 GB cached documents | 10 GB + requests | **$0.25** |
| **Data Transfer** | 5 GB outbound | 5 GB to internet | **$0.45** |
| **Total AWS** | | | **~$26.43/month** |

**Architecture Notes:**
- x86-64 costs ~20% more than ARM64 ($20.83 vs $16.67 for Lambda compute), but provides better stability, compatibility with PyTorch/ONNX, and support for Dockling document parser
- **No API Gateway charges** ($0.10 saved per 100K requests) - Lambda Function URLs are free
- Lambda Function URLs support 15-minute timeout (vs API Gateway's 30-second limit)

**Lambda Free Tier (First 12 Months):**
- 1M requests/month free
- 400,000 GB-seconds/month free
- **Effective cost for 100K requests: $0-5/month** during free tier

### 12.2 External Services (Monthly Estimate)

| Service | Plan | Usage | Monthly Cost |
|---------|------|-------|--------------|
| **OpenAI** | Pay-as-you-go | 100K embeddings + 50K LLM calls | **$10-25** |
| **Pinecone** | Starter | 100K vectors, 2 indexes | **$70** |
| **Supabase** | Free / Pro | 500 MB database | **$0 / $25** |
| **Upstash Redis** | Free / Pay-per-use (optional) | 10K requests/day | **$0 / $10** |
| **OPIK Monitoring** | Free tier (optional) | Basic monitoring | **$0** |
| **Total External** | | | **$80-130/month** |

### 12.3 Total Monthly Cost Estimate

| Scenario | AWS Cost | External Cost | **Total** |
|----------|----------|---------------|-----------|
| **Development** (Free tiers) | $5-10 | $70-100 | **$75-110** |
| **Production** (100K req/month) | $25-27 | $100-130 | **$125-157** |
| **High Traffic** (1M req/month) | $180-200 | $100-130 | **$280-330** |

### 12.4 Cost Optimization Tips

1. ✅ **S3 lifecycle policies** (configured in Section 4.1) - auto-delete old cache
2. ⚡ **Reduce Lambda memory** if possible - lower memory = lower cost
3. 🔄 **Use Upstash Redis caching** - reduce OpenAI API calls
4. 📊 **Monitor with CloudWatch alarms** - catch cost spikes early
5. 🗑️ **Clean up old ECR images** - keep only 3 most recent
6. 📉 **Use Pinecone Serverless** - pay-per-query instead of dedicated pods
7. 🎯 **Enable Dockling** (optional) - USE_DOCKLING=true for better document parsing quality

**Note on ARM64:** While ARM64 Lambda is 20% cheaper, this deployment uses x86-64 for better compatibility with PyTorch, ONNX Runtime, and Dockling. The ~$4-5/month additional cost is worthwhile for production stability.

**Clean up old ECR images:**
```bash
# Keep only 3 most recent images
aws ecr list-images \
  --repository-name rag-text-to-sql \
  --filter tagStatus=TAGGED \
  --query 'sort_by(imageIds, &imagePushedAt)[:-3].imageDigest' \
  --output text | \
  xargs -I {} aws ecr batch-delete-image \
    --repository-name rag-text-to-sql \
    --image-ids imageDigest={}
```

---

## 13. Next Steps

### 13.1 Production Hardening

**Security:**
- [ ] Add API key authentication to endpoints
- [ ] Enable API Gateway access logging
- [ ] Set up AWS WAF for DDoS protection
- [ ] Rotate IAM access keys regularly
- [ ] Enable MFA for AWS account

**Performance:**
- [ ] Enable Provisioned Concurrency for critical endpoints
- [ ] Set up CloudWatch Dashboard for metrics
- [ ] Configure X-Ray tracing for debugging
- [ ] Optimize Lambda memory based on actual usage

**Reliability:**
- [ ] Create staging environment (separate Lambda function)
- [ ] Set up automated backups for Supabase
- [ ] Configure SNS alerts for CloudWatch alarms
- [ ] Document runbook for common incidents

### 13.2 Custom Domain Setup (Optional)

Lambda Function URLs don't natively support custom domains. To use a custom domain:

**Option 1: CloudFront Distribution (Recommended)**
1. **Register domain** in Route 53 (~$12/year for .com)
2. **Request SSL certificate** in ACM for your domain (free)
3. **Create CloudFront distribution**:
   - Origin: Your Lambda Function URL
   - Custom domain: api.yourdomain.com
   - SSL certificate: Your ACM certificate
4. **Update Route 53** CNAME to point to CloudFront
5. **Update GitHub secret** `LAMBDA_FUNCTION_URL` to CloudFront URL

**Option 2: API Gateway Custom Domain (Alternative)**
If you need custom domains and don't mind API Gateway's 30-second timeout:
1. Recreate Section 4.5 with API Gateway HTTP API
2. Follow standard API Gateway custom domain setup
3. Use API Gateway for endpoints < 30 seconds
4. Use Function URL for long-running uploads (>30 seconds)

### 13.3 Advanced Features

**Add Authentication:**
- Implement JWT token validation
- Use AWS Cognito for user management
- Add role-based access control

**Add Rate Limiting:**
- Lambda Function URLs don't have built-in rate limiting
- Use AWS WAF with rate-based rules (costs extra)
- Implement rate limiting in application code
- Alternative: Use API Gateway with usage plans (but loses 15-minute timeout)

**Add Caching:**
- Add CloudFront in front of Function URL for edge caching
- Implement Redis query caching (already supported via Upstash)
- Use CloudFront for CDN capabilities

**Add Observability:**
- Enable X-Ray tracing
- Set up custom CloudWatch metrics
- Integrate with OPIK for LLM observability
- Add structured logging

### 13.4 Team Onboarding

For team members who need to develop locally:

1. **Clone repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/multidata-rag-project.git
   cd multidata-rag-project
   ```

2. **Copy environment template:**
   ```bash
   cp .env.example .env
   # Edit .env with production API keys
   ```

3. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run locally:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access local API:**
   ```
   http://localhost:8000/health
   http://localhost:8000/docs  # Swagger UI
   ```

### 13.5 Monitoring & Alerts Setup

**Create CloudWatch Dashboard:**
1. AWS Console → CloudWatch → Dashboards
2. Create new dashboard: `rag-production`
3. Add widgets:
   - Lambda Invocations (line graph)
   - Lambda Errors (number)
   - Lambda Duration (line graph)
   - API Gateway 4XX/5XX errors
   - S3 bucket size

**Set up SNS topic for alerts:**
```bash
# Create SNS topic
aws sns create-topic --name rag-alerts

# Subscribe email
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:${AWS_ACCOUNT_ID}:rag-alerts \
  --protocol email \
  --notification-endpoint your-email@example.com

# Confirm subscription in email
```

**Update CloudWatch alarms to use SNS:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name rag-lambda-high-errors \
  --alarm-actions arn:aws:sns:us-east-1:${AWS_ACCOUNT_ID}:rag-alerts \
  ... # other parameters
```

---

## Summary

**Congratulations! 🎉** You've successfully deployed a production-ready Multi-Source RAG + Text-to-SQL system on AWS!

**What you built:**
- ✅ Serverless FastAPI application on AWS Lambda (x86-64)
- ✅ Automatic CI/CD with GitHub Actions
- ✅ Document upload and semantic search with Pinecone
- ✅ Natural language SQL queries with Vanna.ai
- ✅ Intelligent query routing (SQL vs Documents vs Hybrid)
- ✅ S3-based document caching for persistence
- ✅ CloudWatch logging and monitoring
- ✅ Lambda Function URL with public HTTPS endpoint (15-minute timeout)

**Your application is:**
- 🚀 **Serverless** - No servers to manage, auto-scales
- 💰 **Cost-effective** - x86-64 architecture, ~$125-157/month for 100K requests (no API Gateway fees)
- 🔄 **Auto-deployed** - Push to main → automatic deployment
- 🔒 **Secure** - HTTPS, IAM roles, encrypted secrets
- 📊 **Observable** - CloudWatch logs, metrics, alarms
- 🔧 **Stable** - Better PyTorch/ONNX compatibility than ARM64
- ⏱️ **Long-running** - 15-minute timeout for large PDF processing (vs API Gateway's 30-second limit)

**Function URL:** `https://[UNIQUE-ID].lambda-url.us-east-1.on.aws/`

**Key Commands:**
```bash
# Test health
curl ${API_URL}/health

# View logs
aws logs tail /aws/lambda/rag-text-to-sql-serverless --follow

# Deploy update
git push origin main

# Rollback
aws lambda update-function-code --function-name rag-text-to-sql --image-uri ${ECR_URI}:PREVIOUS_TAG
```

**Next recommended actions:**
1. Set up custom domain (Section 13.2)
2. Add authentication (Section 13.3)
3. Configure CloudWatch alarms (Section 11.2)
4. Create staging environment
5. Onboard team members (Section 13.4)

**Need help?**
- Check [Troubleshooting](#10-troubleshooting-common-issues)
- View CloudWatch logs
- Review deployment workflow in GitHub Actions
- Refer to AWS documentation: https://docs.aws.amazon.com/lambda/

---

**Document Version:** 3.0
**Last Updated:** 2026-01-25
**Deployment Architecture:** AWS Lambda (x86-64/AMD64) + Lambda Function URL + ECR + S3
**Estimated Setup Time:** 60-90 minutes
**Estimated Monthly Cost:** $125-157 (100K requests/month)
**Architecture Changes:**
- 2026-01-25: Migrated from ARM64 to x86-64 for better PyTorch/ONNX compatibility
- 2026-01-25: Migrated from API Gateway to Lambda Function URL for 15-minute timeout support
