#!/bin/bash
# ============================================================================
# Deployment Script
# ============================================================================
# Deploys the full stack: Lambda + API Gateway + monitoring
# Usage: bash infrastructure/scripts/deploy.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ENV="${ENVIRONMENT:-dev}"
STACK_NAME="product-classifier-${ENV}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Deploying Product Classifier (${ENV})"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Package SAM template
echo "[1/3] Packaging..."
sam package \
    --template-file infrastructure/cloudformation/template.yaml \
    --output-template-file infrastructure/cloudformation/packaged.yaml \
    --s3-bucket "ml-model-artifacts-${ENV}-$(aws sts get-caller-identity --query Account --output text)" \
    --region "$REGION"

# Deploy
echo "[2/3] Deploying stack: ${STACK_NAME}..."
sam deploy \
    --template-file infrastructure/cloudformation/packaged.yaml \
    --stack-name "$STACK_NAME" \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides "Environment=${ENV}" \
    --region "$REGION" \
    --no-fail-on-empty-changeset

# Get outputs
echo "[3/3] Getting stack outputs..."
API_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
    --output text)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Deployment Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  API URL: ${API_URL}"
echo ""
echo "  Test with:"
echo "    curl -X POST ${API_URL} -H 'Content-Type: application/json' -d '{\"image\": \"base64...\"}'"
echo ""
