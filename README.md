# 🏷️ AWS Product Image Classifier

> **Production-grade ML pipeline on AWS** — classifies e-commerce product images into 5 categories with **98.44% accuracy** using a serverless, auto-scaling architecture built entirely on managed AWS services. Trained on **39,110 real product images** from Kaggle's Fashion Product Images dataset.

[![AWS](https://img.shields.io/badge/AWS-Cloud-FF9900?logo=amazonaws)](https://aws.amazon.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions)](/.github/workflows)

---

## 📊 Results

### Key Metrics

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **98.44%** (5 categories) |
| **Test Accuracy** | **98.39%** (3,915 held-out images) |
| Macro F1 Score | 0.9778 |
| Training Accuracy | 94.48% |
| Validation Loss | 0.0496 |
| Dataset Size | **39,110 real product images** |
| Training Images | 31,286 |
| Validation Images | 3,909 |
| Test Images | 3,915 |
| Model Parameters | 11.3M (11.2M trainable) |
| Training Time | ~7 hrs (CPU) / ~35 min (GPU) |
| Best Epoch | 15/15 |
| Inference Latency (p99) | < 800ms |
| Daily Prediction Capacity | 10,000+ (auto-scales to 50K+) |
| Monthly Cost | ~$83 (optimized with spot instances) |

### Training Progression

```
Epoch  1/15 | Train Loss: 0.5979  Train Acc: 79.22% | Val Loss: 0.2047  Val Acc: 94.30% | ★ New best
Epoch  2/15 | Train Loss: 0.4595  Train Acc: 84.74% | Val Loss: 0.1946  Val Acc: 93.66% |
Epoch  3/15 | Train Loss: 0.4093  Train Acc: 86.34% | Val Loss: 0.1292  Val Acc: 95.91% | ★ New best
Epoch  4/15 | Train Loss: 0.3669  Train Acc: 87.71% | Val Loss: 0.1127  Val Acc: 96.34% | ★ New best
Epoch  5/15 | Train Loss: 0.3333  Train Acc: 88.90% | Val Loss: 0.1009  Val Acc: 96.67% | ★ New best
Epoch  6/15 | Train Loss: 0.3008  Train Acc: 89.92% | Val Loss: 0.1227  Val Acc: 96.34% |
Epoch  7/15 | Train Loss: 0.2843  Train Acc: 90.32% | Val Loss: 0.0928  Val Acc: 97.06% | ★ New best
Epoch  8/15 | Train Loss: 0.2604  Train Acc: 91.12% | Val Loss: 0.0764  Val Acc: 97.34% | ★ New best
Epoch  9/15 | Train Loss: 0.2378  Train Acc: 91.98% | Val Loss: 0.0668  Val Acc: 97.57% | ★ New best
Epoch 10/15 | Train Loss: 0.2187  Train Acc: 92.46% | Val Loss: 0.0652  Val Acc: 97.77% | ★ New best
Epoch 11/15 | Train Loss: 0.2025  Train Acc: 93.14% | Val Loss: 0.0616  Val Acc: 98.11% | ★ New best
Epoch 12/15 | Train Loss: 0.1889  Train Acc: 93.47% | Val Loss: 0.0579  Val Acc: 98.11% |
Epoch 13/15 | Train Loss: 0.1754  Train Acc: 93.99% | Val Loss: 0.0542  Val Acc: 98.34% | ★ New best
Epoch 14/15 | Train Loss: 0.1677  Train Acc: 94.29% | Val Loss: 0.0520  Val Acc: 98.39% | ★ New best
Epoch 15/15 | Train Loss: 0.1593  Train Acc: 94.48% | Val Loss: 0.0496  Val Acc: 98.44% | ★ New best
```

### Test Set Evaluation (3,915 held-out images)

```
Overall Accuracy:  98.39%
Macro Precision:   0.9795
Macro Recall:      0.9763
Macro F1 Score:    0.9778
```

| Category | Precision | Recall | F1 Score | Support |
|----------|----------:|-------:|---------:|--------:|
| Apparel | 0.9786 | 0.9898 | 0.9842 | 1,475 |
| Footwear | 0.9978 | 0.9989 | **0.9984** | 923 |
| Accessories | 0.9892 | 0.9818 | 0.9855 | 933 |
| Bottomwear | 0.9706 | 0.9244 | 0.9469 | 357 |
| Personal Care | 0.9614 | 0.9868 | 0.9739 | 227 |

**Confusion Matrix:**

```
                apparel  footwear  access.  bottom.  personal
apparel           1460         0        6        8         1
footwear             1       922        0        0         0
accessories          5         2      916        2         8
bottomwear          26         0        1      330         0
personal_care        0         0        3        0       224
```

**Confidence:** Mean 98.77%, median 99.99%, only 2.02% of predictions below 80% confidence.

### Dataset — Kaggle Fashion Product Images

| Category | Train | Val | Test | Total |
|----------|------:|----:|-----:|------:|
| Apparel (shirts, tops, kurtas) | 11,792 | 1,474 | 1,475 | **14,741** |
| Accessories (watches, bags, belts) | 7,458 | 932 | 933 | **9,323** |
| Footwear (shoes, sandals, heels) | 7,377 | 922 | 923 | **9,222** |
| Bottomwear (jeans, dresses, sarees) | 2,855 | 356 | 357 | **3,568** |
| Personal Care (perfume, makeup) | 1,804 | 225 | 227 | **2,256** |
| **Total** | **31,286** | **3,909** | **3,915** | **39,110** |

> Dataset: [Kaggle Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) — 44,441 professionally shot e-commerce product photos. We mapped 143 article types into 5 visually distinct categories using stratified 80/10/10 splits.

### Model Architecture

```
ResNet18 (pretrained on ImageNet 1.2M images)
├── Backbone: 11.2M parameters (46,528 frozen, rest fine-tuned)
├── Custom head: Dropout(0.5) → Linear(512→256) → ReLU → Dropout(0.25) → Linear(256→5)
├── Optimizer: Adam (lr=0.001, weight_decay=1e-4)
├── Scheduler: Cosine annealing (15 epochs)
├── Augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter, Rotation(±15°)
└── Input: 224×224 RGB, ImageNet-normalized
```

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Client /   │────▶│ API Gateway  │────▶│  Lambda Function │────▶│  SageMaker   │
│   Browser    │     │ (REST API)   │     │  (Preprocess)    │     │  Endpoint    │
└──────────────┘     └──────────────┘     └──────────────────┘     └──────┬───────┘
                                                    │                      │
                                                    ▼                      ▼
                                           ┌──────────────┐     ┌──────────────┐
                                           │  Amazon S3   │     │  DynamoDB    │
                                           │  (Images)    │     │  (Results)   │
                                           └──────────────┘     └──────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                        AUTOMATED RETRAINING PIPELINE                            │
│  EventBridge (Cron) ──▶ Step Functions ──▶ SageMaker Training ──▶ Evaluate     │
│                                                                   ──▶ Deploy   │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                          MONITORING & OBSERVABILITY                              │
│  CloudWatch Dashboards  │  CloudWatch Alarms  │  SNS Alerts  │  Model Monitor  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Architecture Decisions

| Decision | Why | Alternative Considered |
|----------|-----|----------------------|
| **ResNet18** over ResNet50 | 60% fewer params, 2x faster inference, 98.44% accuracy | ResNet50, EfficientNet |
| **Lambda** for preprocessing | Pay-per-request, auto-scales to zero, no idle cost | ECS Fargate |
| **ml.m5.large** for inference | CPU inference 70% cheaper than GPU; ResNet18 fast enough | ml.g4dn.xlarge |
| **DynamoDB** over RDS | Millisecond latency, zero maintenance, pay-per-request | PostgreSQL on RDS |
| **Step Functions** for retraining | Visual debugging, built-in retries, native AWS integrations | Airflow on MWAA |
| **SAM** over raw CloudFormation | Simpler syntax, local testing, built-in Lambda packaging | Terraform, CDK |

---

## 📁 Project Structure

```
aws-product-classifier/
├── README.md
├── prepare_data.py                    # Dataset preparation (Kaggle → train/val/test)
├── Makefile                           # Common commands
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── training_config.yaml           # Hyperparameters & training settings
│   └── infrastructure_config.yaml     # AWS resource configurations
│
├── src/
│   ├── training/
│   │   ├── train.py                   # SageMaker entry point (15-epoch training loop)
│   │   ├── model.py                   # ResNet18 + custom classification head
│   │   ├── dataset.py                 # Custom dataset with augmentation pipeline
│   │   └── evaluate.py                # Confusion matrix, precision/recall/F1
│   │
│   ├── inference/
│   │   ├── inference.py               # SageMaker inference handlers (model_fn → output_fn)
│   │   └── predictor.py               # Client-side prediction helper
│   │
│   ├── preprocessing/
│   │   ├── lambda_handler.py          # Lambda: validate → resize → S3 → SageMaker → DynamoDB
│   │   └── image_utils.py             # EXIF fix, format conversion, validation
│   │
│   ├── pipeline/
│   │   ├── step_functions.json        # Retraining state machine (accuracy-gated deploy)
│   │   ├── data_preparation.py        # Pre-training data validation Lambda
│   │   └── model_evaluation.py        # Post-training evaluation Lambda
│   │
│   ├── monitoring/
│   │   ├── dashboard.py               # CloudWatch dashboard (latency, errors, throughput)
│   │   ├── alarms.py                  # 5 CloudWatch alarms → SNS
│   │   └── model_monitor.py           # SageMaker Model Monitor (data drift detection)
│   │
│   └── utils/
│       ├── aws_helpers.py             # S3, DynamoDB, SageMaker, auto-scaling helpers
│       └── config.py                  # YAML + env var configuration management
│
├── infrastructure/
│   ├── cloudformation/
│   │   ├── template.yaml              # SAM template (Lambda + API Gateway + DynamoDB)
│   │   ├── dynamodb.yaml              # Standalone DynamoDB (DeletionPolicy: Retain)
│   │   └── iam_roles.yaml             # 3 least-privilege IAM roles
│   │
│   └── scripts/
│       ├── setup_environment.sh       # One-click AWS setup (S3, DynamoDB, IAM, SNS)
│       ├── deploy.sh                  # SAM package → deploy → output API URL
│       └── teardown.sh                # Clean resource removal with confirmation
│
├── tests/
│   └── test_model.py                  # 20+ tests: model, preprocessing, Lambda, integration
│
├── notebooks/
│   └── exploration.ipynb              # EDA, augmentation viz, training curves, confusion matrix
│
├── docs/
│   ├── ARCHITECTURE.md                # Detailed architecture with failure mode analysis
│   ├── DEPLOYMENT.md                  # Step-by-step deployment with troubleshooting
│   ├── CONCEPTS.md                    # Every ML & AWS concept explained (interview prep)
│   ├── COST_ANALYSIS.md               # Granular cost modeling per service
│   └── HANDS_ON_GUIDE.md             # Hands-on commands for every AWS service
│
└── .github/
    └── workflows/
        └── ci.yaml                    # GitHub Actions: lint + type check + test + SAM validate
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- AWS CLI v2 configured (`aws configure`)
- [Kaggle dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) downloaded

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/aws-product-classifier.git
cd aws-product-classifier
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision Pillow numpy scikit-learn pyyaml tqdm pytest
```

### 2. Prepare Dataset

```bash
python3 prepare_data.py --source /path/to/kaggle_download --output ./data
```

### 3. Train

```bash
python3 src/training/train.py \
    --train-dir ./data/train \
    --val-dir ./data/val \
    --model-dir ./model \
    --epochs 15 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --patience 5 \
    --num-workers 0
```

### 4. Evaluate

```bash
python3 -c "
from src.training.evaluate import evaluate_from_checkpoint
evaluate_from_checkpoint('./model', './data/test')
"
```

### 5. Deploy to AWS

```bash
make setup          # Creates S3 buckets, DynamoDB, IAM roles
make deploy         # Deploys Lambda + API Gateway + monitoring
```

---

## 🛠️ AWS Services (12)

| Service | Role |
|---------|------|
| **SageMaker** | Model training (spot instances) & real-time inference endpoints |
| **S3** | Image storage with versioning, lifecycle policies, event notifications |
| **Lambda** | Serverless preprocessing (validate, resize, orchestrate) |
| **API Gateway** | REST API with rate limiting (100 req/s), caching, CORS |
| **DynamoDB** | Prediction storage with on-demand scaling, TTL, GSI, streams |
| **Step Functions** | Retraining orchestration with accuracy-gated deployment |
| **EventBridge** | Weekly cron trigger for automated retraining |
| **CloudWatch** | Dashboards, 5 metric alarms, log aggregation |
| **SNS** | Real-time alert notifications on pipeline failures |
| **ECR** | Docker image storage for SageMaker training containers |
| **IAM** | 3 least-privilege roles (SageMaker, Lambda, Step Functions) |
| **CloudFormation/SAM** | Infrastructure as Code for reproducible deployments |

---

## 💰 Cost Analysis

| Service | Usage | Monthly Cost |
|---------|-------|-------------|
| SageMaker Training | 4 hrs/week, ml.g4dn.xlarge (spot) | $13 |
| SageMaker Endpoint | ml.m5.large, 12 hrs/day avg | $41 |
| S3 | 50 GB Standard + 100 GB IA | $2 |
| Lambda | 300K invocations | $6 |
| API Gateway | 300K requests | $1 |
| DynamoDB | 300K writes, 100K reads | $4 |
| CloudWatch | Logs, metrics, 5 alarms | $15 |
| Data Transfer | 10 GB out | $1 |
| **Total** | | **~$83/month** |

---

## 🧪 Testing

```bash
pytest tests/ -v --tb=short              # 20+ unit & integration tests
pytest tests/ -v --cov=src               # With coverage report
```

---

## 📚 Documentation

| Document | Description |
|----------|------------|
| [CONCEPTS.md](docs/CONCEPTS.md) | ML & AWS concepts explained (interview prep) |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Architecture decisions, failure modes, scaling |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Step-by-step deployment guide |
| [COST_ANALYSIS.md](docs/COST_ANALYSIS.md) | Per-service cost modeling |
| [HANDS_ON_GUIDE.md](docs/HANDS_ON_GUIDE.md) | Hands-on CLI commands for every AWS service |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
