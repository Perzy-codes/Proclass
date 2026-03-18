#!/bin/bash
# ============================================================================
# AWS Environment Setup Script
# ============================================================================
# Run once to initialize all AWS resources.
# Usage: bash infrastructure/scripts/setup_environment.sh
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Sufficient IAM permissions

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ENV="${ENVIRONMENT:-dev}"

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  AWS Product Classifier - Environment Setup${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Region:     ${REGION}"
echo "  Account:    ${ACCOUNT_ID}"
echo "  Environment: ${ENV}"
echo ""

# ============================================================================
# Step 1: Create S3 Buckets
# ============================================================================
echo -e "${YELLOW}[1/5] Creating S3 buckets...${NC}"

BUCKETS=(
    "product-images-raw-${ENV}-${ACCOUNT_ID}"
    "product-images-processed-${ENV}-${ACCOUNT_ID}"
    "ml-model-artifacts-${ENV}-${ACCOUNT_ID}"
)

for BUCKET in "${BUCKETS[@]}"; do
    if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
        echo "  ✓ Bucket exists: ${BUCKET}"
    else
        if [ "$REGION" = "us-east-1" ]; then
            aws s3 mb "s3://${BUCKET}"
        else
            aws s3 mb "s3://${BUCKET}" --region "$REGION"
        fi
        echo "  ✓ Created: ${BUCKET}"
    fi
done

# Enable versioning on raw and artifacts buckets
aws s3api put-bucket-versioning \
    --bucket "product-images-raw-${ENV}-${ACCOUNT_ID}" \
    --versioning-configuration Status=Enabled

aws s3api put-bucket-versioning \
    --bucket "ml-model-artifacts-${ENV}-${ACCOUNT_ID}" \
    --versioning-configuration Status=Enabled

echo -e "${GREEN}  S3 buckets ready!${NC}"

# ============================================================================
# Step 2: Create IAM Role for SageMaker
# ============================================================================
echo -e "${YELLOW}[2/5] Setting up IAM roles...${NC}"

ROLE_NAME="SageMakerExecutionRole-${ENV}"

# Check if role exists
if aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
    echo "  ✓ Role exists: ${ROLE_NAME}"
else
    # Create trust policy
    cat > /tmp/trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": [
                    "sagemaker.amazonaws.com",
                    "lambda.amazonaws.com"
                ]
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document file:///tmp/trust-policy.json

    # Attach managed policies
    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

    echo "  ✓ Created role: ${ROLE_NAME}"
fi

echo -e "${GREEN}  IAM roles ready!${NC}"

# ============================================================================
# Step 3: Create DynamoDB Table
# ============================================================================
echo -e "${YELLOW}[3/5] Creating DynamoDB table...${NC}"

TABLE_NAME="product-predictions-${ENV}"

if aws dynamodb describe-table --table-name "$TABLE_NAME" 2>/dev/null; then
    echo "  ✓ Table exists: ${TABLE_NAME}"
else
    aws dynamodb create-table \
        --table-name "$TABLE_NAME" \
        --key-schema \
            AttributeName=product_id,KeyType=HASH \
            AttributeName=timestamp,KeyType=RANGE \
        --attribute-definitions \
            AttributeName=product_id,AttributeType=S \
            AttributeName=timestamp,AttributeType=S \
            AttributeName=category,AttributeType=S \
        --global-secondary-indexes \
            "IndexName=category-index,KeySchema=[{AttributeName=category,KeyType=HASH}],Projection={ProjectionType=ALL}" \
        --billing-mode PAY_PER_REQUEST \
        --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES \
        --region "$REGION"

    echo "  ✓ Created table: ${TABLE_NAME}"
    echo "  Waiting for table to be active..."
    aws dynamodb wait table-exists --table-name "$TABLE_NAME"
fi

echo -e "${GREEN}  DynamoDB table ready!${NC}"

# ============================================================================
# Step 4: Create SNS Topic for Alerts
# ============================================================================
echo -e "${YELLOW}[4/5] Setting up alerting...${NC}"

TOPIC_NAME="ml-pipeline-alerts-${ENV}"
TOPIC_ARN=$(aws sns create-topic --name "$TOPIC_NAME" --query TopicArn --output text)
echo "  ✓ SNS topic: ${TOPIC_ARN}"

if [ -n "${ALERT_EMAIL:-}" ]; then
    aws sns subscribe \
        --topic-arn "$TOPIC_ARN" \
        --protocol email \
        --notification-endpoint "$ALERT_EMAIL"
    echo "  ✓ Subscribed: ${ALERT_EMAIL} (check email to confirm!)"
fi

echo -e "${GREEN}  Alerting ready!${NC}"

# ============================================================================
# Step 5: Create S3 Directory Structure
# ============================================================================
echo -e "${YELLOW}[5/5] Setting up data directory structure...${NC}"

RAW_BUCKET="product-images-raw-${ENV}-${ACCOUNT_ID}"
for SPLIT in train val test; do
    for CATEGORY in apparel footwear accessories bottomwear personal_care; do
        aws s3api put-object \
            --bucket "$RAW_BUCKET" \
            --key "${SPLIT}/${CATEGORY}/" \
            --content-length 0 2>/dev/null || true
    done
done

echo -e "${GREEN}  Directory structure created!${NC}"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Resources created:"
echo "    S3 Buckets:    ${#BUCKETS[@]} buckets"
echo "    IAM Role:      ${ROLE_NAME}"
echo "    DynamoDB:      ${TABLE_NAME}"
echo "    SNS Topic:     ${TOPIC_NAME}"
echo ""
echo "  Next steps:"
echo "    1. Upload training data: aws s3 sync ./data/train s3://${RAW_BUCKET}/train/"
echo "    2. Train model: make train"
echo "    3. Deploy: make deploy"
echo ""
