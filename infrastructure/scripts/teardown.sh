#!/bin/bash
# ============================================================================
# AWS Resource Teardown Script
# ============================================================================
# Removes all AWS resources created by the project.
# Usage: bash infrastructure/scripts/teardown.sh
#
# WARNING: This is destructive! All data will be lost.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ENV="${ENVIRONMENT:-dev}"

echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${RED}  WARNING: This will DELETE all project resources!${NC}"
echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Environment: ${ENV}"
echo "  Region: ${REGION}"
echo ""
read -p "  Type 'DELETE' to confirm: " CONFIRM

if [ "$CONFIRM" != "DELETE" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# Delete SageMaker endpoint
echo -e "${YELLOW}[1/5] Deleting SageMaker endpoint...${NC}"
aws sagemaker delete-endpoint --endpoint-name "product-classifier" 2>/dev/null && \
    echo "  ✓ Endpoint deleted" || echo "  - No endpoint found"

# Delete DynamoDB table
echo -e "${YELLOW}[2/5] Deleting DynamoDB table...${NC}"
aws dynamodb delete-table --table-name "product-predictions-${ENV}" 2>/dev/null && \
    echo "  ✓ Table deleted" || echo "  - No table found"

# Empty and delete S3 buckets
echo -e "${YELLOW}[3/5] Deleting S3 buckets...${NC}"
for BUCKET in \
    "product-images-raw-${ENV}-${ACCOUNT_ID}" \
    "product-images-processed-${ENV}-${ACCOUNT_ID}" \
    "ml-model-artifacts-${ENV}-${ACCOUNT_ID}"; do
    
    if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
        aws s3 rb "s3://${BUCKET}" --force 2>/dev/null && \
            echo "  ✓ Deleted: ${BUCKET}" || echo "  ! Failed: ${BUCKET}"
    else
        echo "  - Not found: ${BUCKET}"
    fi
done

# Delete CloudFormation stack
echo -e "${YELLOW}[4/5] Deleting CloudFormation stack...${NC}"
aws cloudformation delete-stack --stack-name "product-classifier-${ENV}" 2>/dev/null && \
    echo "  ✓ Stack deletion initiated" || echo "  - No stack found"

# Delete CloudWatch resources
echo -e "${YELLOW}[5/5] Cleaning up monitoring...${NC}"
for ALARM in \
    "ProductClassifier-HighLatency" \
    "ProductClassifier-HighErrorRate" \
    "ProductClassifier-LambdaThrottles" \
    "ProductClassifier-API5xxErrors" \
    "ProductClassifier-EndpointErrors"; do
    aws cloudwatch delete-alarms --alarm-names "$ALARM" 2>/dev/null
done
echo "  ✓ Alarms deleted"

aws cloudwatch delete-dashboards --dashboard-names "ProductClassifierDashboard" 2>/dev/null && \
    echo "  ✓ Dashboard deleted" || echo "  - No dashboard found"

echo ""
echo -e "${GREEN}Teardown complete!${NC}"
