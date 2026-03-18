# Deployment Guide

Step-by-step instructions for deploying the Product Classifier to AWS.

---

## Prerequisites

1. **AWS Account** with admin access (or sufficient IAM permissions)
2. **AWS CLI v2** installed and configured
3. **Python 3.10+** with pip
4. **Docker** (for local testing and ECR push)
5. **AWS SAM CLI** (for CloudFormation deployment)

```bash
# Verify prerequisites
aws --version          # Should be 2.x
python --version       # Should be 3.10+
docker --version       # Should be 20+
sam --version          # Should be 1.x
```

---

## Step 1: Clone & Configure

```bash
git clone https://github.com/yourusername/aws-product-classifier.git
cd aws-product-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your AWS account ID and preferences
```

---

## Step 2: Setup AWS Infrastructure

```bash
# Option A: Automated setup (recommended)
make setup

# Option B: Manual setup
bash infrastructure/scripts/setup_environment.sh
```

This creates S3 buckets (raw, processed, artifacts), a DynamoDB table, IAM roles, and an SNS alert topic.

**Verify:**
```bash
aws s3 ls | grep product-images    # Should list 2 buckets
aws dynamodb list-tables           # Should include product-predictions-dev
```

---

## Step 3: Prepare Training Data

Upload your labeled product images to S3:

```bash
# Expected structure:
# data/
#   train/
#     electronics/  (1600+ images)
#     clothing/     (1600+ images)
#     furniture/    (1600+ images)
#     books/        (1600+ images)
#     toys/         (1600+ images)
#   val/
#     (same categories, ~200 images each)
#   test/
#     (same categories, ~200 images each)

# Upload to S3
aws s3 sync data/train/ s3://product-images-raw-dev-YOUR_ACCOUNT_ID/train/
aws s3 sync data/val/   s3://product-images-raw-dev-YOUR_ACCOUNT_ID/val/
aws s3 sync data/test/  s3://product-images-raw-dev-YOUR_ACCOUNT_ID/test/
```

**No data yet?** Use the Fashion-MNIST dataset or scrape product images from public datasets. The notebooks/ folder has an exploration notebook for dataset preparation.

---

## Step 4: Train the Model

```bash
# Option A: Train on SageMaker (recommended for production)
make train

# Option B: Train locally (for development)
make train-local
```

SageMaker training takes about 35 minutes on ml.g4dn.xlarge. Monitor progress in the SageMaker console or CloudWatch logs.

---

## Step 5: Deploy the Endpoint

```bash
# Deploy the model to a SageMaker endpoint
make deploy-model

# Or deploy the full stack (Lambda + API Gateway + monitoring)
make deploy
```

---

## Step 6: Test the Deployment

```bash
# Health check
curl https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/dev/health

# Classify an image
python src/inference/predictor.py --image test_product.jpg --api-url https://YOUR_API_URL
```

---

## Step 7: Setup Monitoring

```bash
make monitoring-setup
```

Check the CloudWatch dashboard in the AWS Console for real-time metrics.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SageMaker training fails | Check CloudWatch logs for the training job |
| Endpoint returns 500 | Check endpoint CloudWatch logs; verify model artifact exists in S3 |
| Lambda timeout | Increase timeout in SAM template (max 900s) |
| High latency | Check if auto-scaling is configured; consider GPU inference instance |
| Permission denied | Verify IAM role has required policies attached |

---

## Cost Management

After testing, tear down resources to avoid charges:

```bash
make teardown
```

For ongoing use, verify auto-scaling is configured to minimize idle costs.
