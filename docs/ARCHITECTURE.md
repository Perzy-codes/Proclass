# Architecture Documentation

Detailed technical architecture for the Product Image Classifier.

---

## System Overview

The system is designed around four pillars: **ingest**, **predict**, **retrain**, and **observe**.

```
                    ┌─────────────────────────────────────────┐
                    │           INGEST LAYER                   │
                    │                                         │
   Client ────────▶│  API Gateway ──▶ Lambda ──▶ S3          │
                    │     (rate limit)   (resize)  (store)    │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │          PREDICT LAYER                   │
                    │                                         │
                    │  SageMaker Endpoint (auto-scaling 1-5)  │
                    │  ResNet18 + Custom Head                 │
                    │  ──▶ DynamoDB (store result)            │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │          RETRAIN LAYER                   │
                    │                                         │
                    │  EventBridge (weekly cron)               │
                    │  ──▶ Step Functions                      │
                    │       ├── PrepareData (Lambda)           │
                    │       ├── TrainModel (SageMaker)         │
                    │       ├── Evaluate (Lambda)              │
                    │       └── Deploy if accuracy ≥ 90%       │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │          OBSERVE LAYER                   │
                    │                                         │
                    │  CloudWatch (metrics, logs, dashboards)  │
                    │  CloudWatch Alarms ──▶ SNS ──▶ Email    │
                    │  SageMaker Model Monitor (data drift)    │
                    └─────────────────────────────────────────┘
```

---

## Request Flow (Real-Time Prediction)

```
1. Client sends POST /classify with base64 image
   │
2. API Gateway validates request, applies rate limiting
   │
3. Lambda function:
   ├── Decodes base64 image
   ├── Validates format and size
   ├── Resizes to 224×224
   ├── Stores processed image in S3 (audit trail)
   └── Calls SageMaker endpoint
       │
4. SageMaker endpoint:
   ├── input_fn: Tensor conversion + normalization
   ├── predict_fn: Model forward pass (ResNet18)
   └── output_fn: JSON serialization with probabilities
   │
5. Lambda stores result in DynamoDB
   │
6. Response returned through API Gateway to client
   
Total latency: ~200-800ms (p99)
```

---

## Design Decisions

### Why Serverless-First?

We chose serverless (Lambda + API Gateway) over containers (ECS/EKS) because:
- Zero cost when idle (Lambda charges per invocation)
- Auto-scaling is built-in (no cluster management)
- Faster iteration (deploy in seconds, not minutes)
- Lower operational burden (no patching, no capacity planning)

The tradeoff is cold starts (~1-3 seconds on first invocation), which is acceptable for this use case.

### Why SageMaker over Lambda for Inference?

Lambda has a 10 GB deployment size limit. PyTorch + model weights exceed this. Also, SageMaker keeps the model in memory permanently, avoiding cold start overhead for the ML portion.

### Why DynamoDB over RDS?

Prediction results are simple key-value lookups (get predictions for product X). DynamoDB gives millisecond latency for this pattern without any database management.

We chose on-demand billing over provisioned capacity because prediction traffic is bursty and unpredictable.

### Why ResNet18 over Larger Models?

For 5 classes with ~10K training images, ResNet18 provides 92%+ accuracy with 2x faster inference than ResNet50 at a fraction of the cost. The accuracy difference is negligible (<1%) for this task size.

---

## Security Architecture

### IAM Least Privilege

Each component gets only the permissions it needs:

| Component | Permissions |
|-----------|------------|
| Lambda | S3 PutObject, SageMaker InvokeEndpoint, DynamoDB PutItem |
| SageMaker Training | S3 Read (data), S3 Write (artifacts), CloudWatch Logs |
| SageMaker Endpoint | S3 Read (model), CloudWatch Logs |
| API Gateway | Lambda Invoke |
| Step Functions | Lambda Invoke, SageMaker CreateTrainingJob, SNS Publish |

### Data Protection

- S3 buckets block all public access
- S3 versioning protects against accidental deletion
- DynamoDB encryption at rest (default AES-256)
- API Gateway can enforce API key authentication
- All inter-service communication uses IAM roles (no hardcoded credentials)

---

## Failure Modes & Mitigations

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| SageMaker endpoint down | No predictions | CloudWatch alarm → SNS alert. Auto-scaling replaces unhealthy instances. |
| Lambda error | Individual request fails | Retry logic in client. CloudWatch alarm on error rate. |
| DynamoDB throttling | Predictions not stored | On-demand billing auto-scales. Alarm on throttled requests. |
| Model accuracy degrades | Bad predictions | Model Monitor detects drift. Weekly retraining pipeline. |
| Training fails | No new model | Step Functions retry. SNS notification. Previous model stays active. |
| S3 outage | Can't store/retrieve images | Multi-AZ by default. Extremely rare (11 nines durability). |

---

## Scaling Strategy

| Load | Endpoint | Lambda | DynamoDB |
|------|----------|--------|----------|
| Low (<100/min) | 1 instance | Default concurrency | On-demand |
| Medium (<1K/min) | 2-3 instances (auto-scale) | Default | On-demand |
| High (<10K/min) | 3-5 instances | Reserved concurrency | On-demand |
| Burst (>10K/min) | Consider batch endpoint | Request reserved concurrency increase | On-demand auto-scales |
