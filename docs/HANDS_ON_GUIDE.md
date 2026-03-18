# Hands-On Guide — Run This Project From Scratch

Step-by-step instructions to train the model and use every AWS service.

---

## Dataset

**Kaggle Fashion Product Images (Small)**  
- 39,114 real e-commerce product images  
- 5 categories: Apparel, Footwear, Accessories, Bottomwear, Personal Care  
- Source: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

---

## Phase 1: Local Training ($0)

### Step 1: Download the Dataset

```bash
# Option A: Kaggle CLI (recommended)
pip install kaggle
kaggle datasets download -d paramaggarwal/fashion-product-images-small
unzip fashion-product-images-small.zip -d kaggle_data/

# Option B: Manual download from Kaggle website
# Go to the URL above → Download → Unzip
```

### Step 2: Prepare the Data

```bash
cd aws-product-classifier

# This maps 44K Kaggle images into our 5 categories + train/val/test splits
python prepare_data.py --source /path/to/kaggle_data --output ./data

# Expected output:
#   train/
#     apparel/       ~11,795 images
#     footwear/       ~7,377 images
#     accessories/    ~7,458 images
#     bottomwear/     ~2,855 images
#     personal_care/  ~1,804 images
#   val/    (~10% of each)
#   test/   (~10% of each)
```

### Step 3: Train Locally

```bash
# Full training (CPU: ~2-3 hours, GPU: ~15-20 min)
python src/training/train.py \
    --train-dir ./data/train \
    --val-dir ./data/val \
    --model-dir ./model \
    --epochs 15 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --patience 5

# Quick test run (5 min, to verify everything works)
python src/training/train.py \
    --train-dir ./data/train \
    --val-dir ./data/val \
    --model-dir ./model \
    --epochs 2 \
    --batch-size 64 \
    --num-workers 2
```

**CAPTURE THIS OUTPUT** — copy-paste the epoch logs for your README.

### Step 4: Evaluate

```bash
python -c "
from src.training.evaluate import evaluate_from_checkpoint
evaluate_from_checkpoint('./model', './data/test')
"
```

**SCREENSHOT THE OUTPUT** — the confusion matrix and per-class metrics go in your README.

### Step 5: Test Prediction

```bash
# Classify a single image
python src/inference/predictor.py --image ./data/test/footwear/12345.jpg --local --model-dir ./model
```

---

## Phase 2: AWS Free Tier Services ($0)

### Prerequisites
```bash
# Install AWS CLI
pip install awscli boto3

# Configure (use your AWS credentials)
aws configure
# Region: us-east-1
```

### Service 1: Amazon S3

```bash
# Create buckets
aws s3 mb s3://product-images-raw-dev-YOUR_ACCOUNT_ID
aws s3 mb s3://product-images-processed-dev-YOUR_ACCOUNT_ID
aws s3 mb s3://ml-model-artifacts-dev-YOUR_ACCOUNT_ID

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket product-images-raw-dev-YOUR_ACCOUNT_ID \
    --versioning-configuration Status=Enabled

# Upload a few test images
aws s3 cp ./data/test/apparel/ s3://product-images-raw-dev-YOUR_ACCOUNT_ID/test/apparel/ --recursive --max-items 10

# List uploaded files
aws s3 ls s3://product-images-raw-dev-YOUR_ACCOUNT_ID/test/apparel/ --human-readable

# Set lifecycle policy (move to cheaper storage after 30 days)
aws s3api put-bucket-lifecycle-configuration \
    --bucket product-images-raw-dev-YOUR_ACCOUNT_ID \
    --lifecycle-configuration '{
        "Rules": [{
            "ID": "MoveToIA",
            "Status": "Enabled",
            "Transitions": [{"Days": 30, "StorageClass": "STANDARD_IA"}],
            "Filter": {"Prefix": ""}
        }]
    }'

# INTERVIEW PREP: Now you can talk about S3 versioning, lifecycle policies,
# storage classes, and event notifications from real experience.
```

### Service 2: DynamoDB

```bash
# Create table
aws dynamodb create-table \
    --table-name product-predictions-dev \
    --key-schema \
        AttributeName=product_id,KeyType=HASH \
        AttributeName=timestamp,KeyType=RANGE \
    --attribute-definitions \
        AttributeName=product_id,AttributeType=S \
        AttributeName=timestamp,AttributeType=S \
    --billing-mode PAY_PER_REQUEST

# Wait for table to be active
aws dynamodb wait table-exists --table-name product-predictions-dev

# Insert a test prediction
aws dynamodb put-item \
    --table-name product-predictions-dev \
    --item '{
        "product_id": {"S": "TEST-001"},
        "timestamp": {"S": "1700000000000"},
        "category": {"S": "footwear"},
        "confidence": {"N": "0.94"},
        "request_id": {"S": "req-abc-123"}
    }'

# Query it back
aws dynamodb get-item \
    --table-name product-predictions-dev \
    --key '{"product_id": {"S": "TEST-001"}, "timestamp": {"S": "1700000000000"}}'

# Scan all items
aws dynamodb scan --table-name product-predictions-dev

# Enable TTL (auto-delete old items)
aws dynamodb update-time-to-live \
    --table-name product-predictions-dev \
    --time-to-live-specification Enabled=true,AttributeName=ttl

# INTERVIEW PREP: You can now discuss partition/sort keys, on-demand billing,
# GSIs, TTL, and DynamoDB Streams from real experience.
```

### Service 3: AWS Lambda

```bash
# Create a simple test Lambda (zip the handler)
cd src/preprocessing
zip -r /tmp/lambda_function.zip lambda_handler.py image_utils.py

# Create the function
aws lambda create-function \
    --function-name product-classifier-test \
    --runtime python3.10 \
    --handler lambda_handler.lambda_handler \
    --zip-file fileb:///tmp/lambda_function.zip \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/LambdaBasicRole \
    --timeout 30 \
    --memory-size 1024

# Invoke with a test event
aws lambda invoke \
    --function-name product-classifier-test \
    --payload '{"body": "{\"test\": true}"}' \
    /tmp/lambda_response.json

cat /tmp/lambda_response.json

# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/product-classifier

# INTERVIEW PREP: Cold starts, concurrency, memory/CPU tradeoff, event-driven.
```

### Service 4: API Gateway

```bash
# Create REST API
aws apigateway create-rest-api \
    --name ProductClassifierAPI \
    --description "Product image classification API"

# Note the API ID from the output, then:
# Create resource, method, integration, deploy stage
# (Or use SAM template which does all this automatically:)
# sam deploy --guided --template infrastructure/cloudformation/template.yaml
```

### Service 5: IAM Roles

```bash
# View the role we created
aws iam get-role --role-name SageMakerExecutionRole-dev

# List attached policies
aws iam list-attached-role-policies --role-name SageMakerExecutionRole-dev

# INTERVIEW PREP: Least privilege, service-linked roles, trust policies.
```

### Service 6: CloudWatch

```bash
# Create a dashboard
python src/monitoring/dashboard.py

# Manually create a test alarm
aws cloudwatch put-metric-alarm \
    --alarm-name TestAlarm \
    --metric-name Errors \
    --namespace AWS/Lambda \
    --statistic Sum \
    --period 300 \
    --evaluation-periods 1 \
    --threshold 5 \
    --comparison-operator GreaterThanThreshold

# Test the alarm
aws cloudwatch set-alarm-state \
    --alarm-name TestAlarm \
    --state-value ALARM \
    --state-reason "Testing"

# Check alarm state
aws cloudwatch describe-alarms --alarm-names TestAlarm

# INTERVIEW PREP: Metrics, logs, alarms, dashboards, Logs Insights.
```

### Service 7: SNS

```bash
# Create topic
aws sns create-topic --name ml-alerts-test

# Subscribe your email
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-1:YOUR_ACCOUNT:ml-alerts-test \
    --protocol email \
    --notification-endpoint your@email.com

# Publish a test message
aws sns publish \
    --topic-arn arn:aws:sns:us-east-1:YOUR_ACCOUNT:ml-alerts-test \
    --subject "Test Alert" \
    --message "This is a test alert from the ML pipeline"

# Check your email!
```

### Service 8: EventBridge

```bash
# Create a scheduled rule (won't cost anything unless it triggers resources)
aws events put-rule \
    --name test-weekly-schedule \
    --schedule-expression 'cron(0 2 ? * SUN *)' \
    --state DISABLED

# View the rule
aws events describe-rule --name test-weekly-schedule

# INTERVIEW PREP: Cron expressions, event patterns, targets.
```

---

## Phase 3: SageMaker (~$5-8, run once)

### Step 1: Upload Training Data to S3

```bash
# Upload the full training dataset
aws s3 sync ./data/train/ s3://product-images-raw-dev-YOUR_ACCOUNT_ID/train/ 
aws s3 sync ./data/val/ s3://product-images-raw-dev-YOUR_ACCOUNT_ID/val/
aws s3 sync ./data/test/ s3://product-images-raw-dev-YOUR_ACCOUNT_ID/test/

# Verify upload
aws s3 ls s3://product-images-raw-dev-YOUR_ACCOUNT_ID/train/ --recursive --summarize
```

### Step 2: Launch SageMaker Training Job (~$2-3)

```python
# Run in Python or Jupyter notebook
import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole-dev'

estimator = PyTorch(
    entry_point='train.py',
    source_dir='src/training/',
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    use_spot_instances=True,          # Save 70%!
    max_wait=7200,
    max_run=3600,
    hyperparameters={
        'epochs': 15,
        'batch-size': 32,
        'learning-rate': 0.001,
        'patience': 5,
    }
)

estimator.fit({
    'train': f's3://product-images-raw-dev-YOUR_ACCOUNT_ID/train/',
    'validation': f's3://product-images-raw-dev-YOUR_ACCOUNT_ID/val/',
})

print(f"Model artifacts: {estimator.model_data}")
# SAVE THIS S3 URI — you need it for deployment
```

### Step 3: Deploy Endpoint (~$2-3 for 1-2 hours)

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data=estimator.model_data,  # From step 2
    role=role,
    entry_point='inference.py',
    source_dir='src/inference/',
    framework_version='2.0.0',
    py_version='py310',
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='product-classifier',
)

# Test the endpoint
import json
with open('data/test/footwear/12345.jpg', 'rb') as f:
    response = predictor.predict(f.read(), initial_args={'ContentType': 'application/x-image'})
print(json.loads(response))

# SCREENSHOT THIS — real prediction from deployed endpoint
```

### Step 4: TEAR DOWN IMMEDIATELY

```bash
# Delete endpoint (stops billing!)
aws sagemaker delete-endpoint --endpoint-name product-classifier
aws sagemaker delete-endpoint-config --endpoint-config-name product-classifier

# Verify it's deleted
aws sagemaker list-endpoints
```

---

## Phase 4: Capture Results for README

Take screenshots / copy output of:
1. Training epoch logs (loss decreasing, accuracy increasing)
2. Evaluation report (confusion matrix, per-class precision/recall/F1)
3. Sample prediction output (image → category + confidence)
4. CloudWatch dashboard (if you set one up)
5. SageMaker training job in the console
6. API response from curl (if you deployed the full stack)

These go in a `Results` section in your README — proof the project actually runs.

---

## Cleanup Checklist

After capturing all screenshots:

```bash
# Delete SageMaker endpoint (most expensive)
aws sagemaker delete-endpoint --endpoint-name product-classifier 2>/dev/null

# Delete Lambda
aws lambda delete-function --function-name product-classifier-test 2>/dev/null

# Delete DynamoDB table
aws dynamodb delete-table --table-name product-predictions-dev 2>/dev/null

# Delete CloudWatch alarms
aws cloudwatch delete-alarms --alarm-names TestAlarm 2>/dev/null

# Delete SNS topic
aws sns delete-topic --topic-arn arn:aws:sns:us-east-1:YOUR_ACCOUNT:ml-alerts-test 2>/dev/null

# Delete EventBridge rule
aws events delete-rule --name test-weekly-schedule 2>/dev/null

# S3 buckets (optional — keep if you want to re-run later)
# aws s3 rb s3://product-images-raw-dev-YOUR_ACCOUNT_ID --force

echo "All resources cleaned up!"
```
