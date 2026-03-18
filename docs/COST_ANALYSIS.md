# Cost Analysis & Optimization

Detailed cost breakdown and optimization strategies.

---

## Monthly Cost Breakdown (~$128/month)

### Compute Costs

| Service | Configuration | Hours/Month | Rate | Monthly Cost |
|---------|--------------|-------------|------|-------------|
| SageMaker Training | ml.g4dn.xlarge (spot) | 16 hrs | $0.526/hr × 0.3 (spot) | **$13** |
| SageMaker Endpoint | ml.m5.large | ~360 hrs (12hr/day avg) | $0.115/hr | **$41** |
| Lambda | 300K invocations, 1024MB, avg 2s | — | $0.0000166667/GB-s | **$6** |

### Storage Costs

| Service | Usage | Rate | Monthly Cost |
|---------|-------|------|-------------|
| S3 Standard | 50 GB (recent images) | $0.023/GB | **$1.15** |
| S3 Infrequent Access | 100 GB (older images) | $0.0125/GB | **$1.25** |
| DynamoDB (on-demand) | 300K writes, 100K reads | $1.25/M writes, $0.25/M reads | **$0.40** |

### API & Networking

| Service | Usage | Rate | Monthly Cost |
|---------|-------|------|-------------|
| API Gateway | 300K requests | $3.50/M requests | **$1.05** |
| Data Transfer | 10 GB outbound | $0.09/GB | **$0.90** |

### Monitoring

| Service | Usage | Monthly Cost |
|---------|-------|-------------|
| CloudWatch Logs | 5 GB ingestion | **$2.50** |
| CloudWatch Metrics | 20 custom metrics | **$6.00** |
| CloudWatch Alarms | 5 alarms | **$0.50** |
| CloudWatch Dashboard | 1 dashboard | **$3.00** |

### Total: ~$77/month (optimized) to ~$128/month (standard)

---

## Optimization Strategies

### 1. Spot Instances for Training (saves ~$29/month)

Spot instances use spare AWS capacity at 60-70% discount. SageMaker handles interruptions by checkpointing and resuming automatically.

```python
estimator = PyTorch(
    use_spot_instances=True,          # Enable spot
    max_wait=7200,                    # Max wait for spot capacity
    checkpoint_s3_uri='s3://bucket/checkpoints/',  # Auto-resume
)
```

### 2. Endpoint Auto-Scaling (saves ~$20-40/month)

Scale to fewer instances during low-traffic hours. Configure scaling to zero during nights/weekends if acceptable.

### 3. S3 Intelligent-Tiering (saves ~$0.50/month)

Automatically moves objects between access tiers based on usage patterns. No retrieval fees, small monitoring fee ($0.0025/1000 objects).

### 4. API Gateway Response Caching (saves ~$2/month)

Cache responses for identical requests. If users upload the same product image twice, return the cached result instead of invoking Lambda + SageMaker again.

### 5. Reserved Capacity (saves ~20% on steady-state)

If running 24/7 in production, consider SageMaker Savings Plans for the endpoint instance.

---

## Scaling Cost Projections

| Daily Predictions | Endpoint Config | Lambda | DynamoDB | Est. Monthly |
|-------------------|----------------|--------|----------|-------------|
| 1,000 | 1x ml.m5.large | $2 | $0.10 | **~$55** |
| 10,000 | 1x ml.m5.large | $6 | $0.40 | **~$77** |
| 50,000 | 2x ml.m5.large | $30 | $2.00 | **~$160** |
| 100,000 | 3x ml.m5.large | $60 | $4.00 | **~$270** |
| 500,000 | 5x ml.m5.large | $300 | $20.00 | **~$750** |

These are estimates. Actual costs vary by region, usage patterns, and optimization level.
