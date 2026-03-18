# ML & AWS Concepts — Interview Prep Guide

Everything you need to understand about this project, explained from the ground up.

---

## Table of Contents

1. [Machine Learning Fundamentals](#1-machine-learning-fundamentals)
2. [Neural Networks & CNNs](#2-neural-networks--cnns)
3. [Transfer Learning Deep Dive](#3-transfer-learning-deep-dive)
4. [Training Pipeline Concepts](#4-training-pipeline-concepts)
5. [AWS Services Explained](#5-aws-services-explained)
6. [MLOps & Production ML](#6-mlops--production-ml)
7. [System Design Interview Questions](#7-system-design-interview-questions)

---

## 1. Machine Learning Fundamentals

### What is Machine Learning?

Machine learning is teaching computers to make decisions by showing them examples instead of writing explicit rules.

**Traditional Programming:**
```
Input + Rules → Output
"If image has rectangular shape AND has pages → classify as book"
```

**Machine Learning:**
```
Input + Output → Rules (learned automatically)
"Here are 10,000 labeled images" → Model learns the patterns
```

### Supervised Learning

Our project uses **supervised learning** — we provide labeled examples (image + correct category) and the model learns the mapping.

**Training data looks like:**
```
laptop.jpg    → "electronics"
tshirt.jpg    → "clothing"
sofa.jpg      → "furniture"
novel.jpg     → "books"
lego.jpg      → "toys"
```

The model learns patterns like:
- Electronics often have screens, buttons, metallic surfaces
- Clothing has fabric textures, varied colors, human shapes
- Furniture is large, has wooden/fabric textures, room context

### The ML Workflow

```
1. Collect Data
      │
2. Prepare Data (clean, split, augment)
      │
3. Choose Model Architecture
      │
4. Train Model (forward pass → loss → backward pass → update weights)
      │
5. Evaluate Model (accuracy, precision, recall on held-out test set)
      │
6. Deploy Model (SageMaker endpoint)
      │
7. Monitor & Retrain (drift detection, weekly retraining)
```

### Train / Validation / Test Split

We split our data into three sets:

| Set | % | Purpose | Used During |
|-----|---|---------|-------------|
| **Train** | 80% | Model learns from these | Training |
| **Validation** | 10% | Tune hyperparameters, detect overfitting | Training |
| **Test** | 10% | Final unbiased evaluation | After training |

**Why three sets?**
- If we only had train/test, we'd tune hyperparameters on the test set, which biases our evaluation
- The test set must be "unseen" — touched only ONCE for final evaluation
- This gives an honest estimate of real-world performance

### Overfitting vs. Underfitting

**Overfitting:** Model memorizes training data but fails on new data.
- Analogy: A student who memorizes answers to practice tests but can't solve new problems
- Signs: Training accuracy 99%, validation accuracy 70%
- Solutions: More data, dropout, data augmentation, simpler model, early stopping

**Underfitting:** Model is too simple to capture patterns.
- Analogy: A student who didn't study enough
- Signs: Both training AND validation accuracy are low (60%)
- Solutions: More complex model, train longer, better features, lower regularization

---

## 2. Neural Networks & CNNs

### What is a Neural Network?

A neural network is a function that transforms input data through layers of learned parameters.

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer
   (image)     (edges)          (textures)              (category)
```

Each layer applies: `output = activation(weights × input + bias)`

**Key components:**
- **Weights:** Learnable parameters (multiplied with input)
- **Bias:** Learnable offset (added after multiplication)
- **Activation function:** Non-linearity (ReLU, softmax, etc.)

### Convolutional Neural Networks (CNNs)

CNNs are specifically designed for image data. Instead of connecting every pixel to every neuron (computationally impossible), they use **convolutions** — sliding filters that detect local patterns.

```
Convolution: A small filter (e.g., 3×3) slides across the image.
At each position, it computes: sum(filter × image_patch)

Filter for detecting vertical edges:
[-1  0  1]
[-1  0  1]
[-1  0  1]

This filter produces a high value where there's a vertical edge
and a low value where there isn't.
```

**Why CNNs work for images:**
1. **Local patterns:** A filter detects the same pattern anywhere in the image
2. **Parameter sharing:** One 3×3 filter = 9 parameters, regardless of image size
3. **Hierarchical features:** Stack filters to detect increasingly complex features

### The CNN Feature Hierarchy

```
Layer 1: Edges, gradients, color blobs
         ┌─┐  ╱  ━
Layer 2: Textures, corners, simple shapes
         ┌──┐  ╲╱  ◻
Layer 3: Object parts (buttons, fabric, pages)
         📱  👕  📖
Layer 4: Full objects (laptop, dress, bookshelf)
         💻  👗  📚
```

Early layers are UNIVERSAL (edges are edges in any image).
Later layers are TASK-SPECIFIC (what makes a "laptop" vs. a "book").

### ResNet Architecture

ResNet's key innovation: **skip connections** (residual connections).

```
Standard block:
  Input → Conv → BN → ReLU → Conv → BN → ReLU → Output

ResNet block:
  Input → Conv → BN → ReLU → Conv → BN → (+) → ReLU → Output
    │                                      ↑
    └──────────────────────────────────────┘
                  (skip connection)

Output = F(Input) + Input

If F can't learn anything useful: F(Input) ≈ 0, so Output ≈ Input
The layer just passes data through — no harm done!
```

This allows training very deep networks (18, 50, 101, 152 layers).

---

## 3. Transfer Learning Deep Dive

### The Concept

Transfer learning reuses knowledge from one task to solve another.

```
Source Task: ImageNet (1.2M images, 1000 classes)
  ↓ Transfer knowledge (keep learned features)
Target Task: Product Classification (10K images, 5 classes)
```

### Why It Works

ResNet18 trained on ImageNet has already learned to "see." The early layers detect universal visual features that apply to ANY image task. We only need to teach it our specific categories.

**Without transfer learning:** Need 100K+ images per class, weeks of training
**With transfer learning:** Need ~2K images per class, ~30 minutes of training

### Fine-Tuning Strategies

| Strategy | What's Trained | When to Use |
|----------|---------------|-------------|
| **Feature extraction** | Only new FC layer | Very small dataset (<500/class) |
| **Partial fine-tuning** | Last few layers + FC | Medium dataset (our approach) |
| **Full fine-tuning** | All layers | Large dataset (10K+/class) |

We use **partial fine-tuning**: freeze Layers 1-2 (universal), fine-tune Layers 3-4 + FC (task-specific).

### ImageNet Normalization

Pre-trained ResNet expects input normalized with ImageNet statistics:
```python
mean = [0.485, 0.456, 0.406]  # RGB means
std  = [0.229, 0.224, 0.225]  # RGB standard deviations
```

**If you forget normalization:** The model receives input on a different scale than it was trained on. Accuracy drops from 92% to ~30%. This is the #1 deployment bug.

---

## 4. Training Pipeline Concepts

### The Training Loop

Every deep learning training loop has exactly 5 critical lines:

```python
optimizer.zero_grad()      # 1. Clear old gradients
outputs = model(inputs)    # 2. Forward pass
loss = criterion(outputs, labels)  # 3. Compute loss
loss.backward()            # 4. Backpropagation
optimizer.step()           # 5. Update weights
```

### Loss Functions

**CrossEntropyLoss** (what we use):
```
For true label "electronics" (class 0):
  Model outputs: [2.1, 0.5, -1.0, 0.3, -0.8]
  After softmax: [0.55, 0.11, 0.02, 0.09, 0.03]
  Loss = -log(0.55) = 0.60  (low loss — good!)

For true label "electronics" with bad prediction:
  Model outputs: [-1.0, 2.5, 0.3, 0.1, -0.5]
  After softmax: [0.02, 0.72, 0.08, 0.07, 0.04]
  Loss = -log(0.02) = 3.91  (high loss — bad!)
```

The loss is ALWAYS a single number. Gradient descent minimizes this number.

### Optimizers

**SGD (Stochastic Gradient Descent):**
```
w = w - learning_rate × gradient
```
Simple but slow, sensitive to learning rate.

**Adam (what we use):**
```
Maintains per-parameter adaptive learning rates.
Combines momentum (smooths gradients) + RMSprop (adapts step size).
Works well with default settings (lr=0.001).
```

### Learning Rate

The learning rate (LR) controls how big of a step we take during optimization.

```
Too high (LR=1.0):    Overshoots, loss explodes  📈💥
Too low (LR=0.00001): Barely moves, takes forever  🐌
Just right (LR=0.001): Converges smoothly            ✓

We use cosine annealing:
  Start: LR=0.001 (explore broadly)
  End:   LR=0.00001 (fine-tune precisely)
```

### Regularization Techniques

| Technique | What It Does | When to Use |
|-----------|-------------|-------------|
| **Dropout** | Randomly zeros neurons during training | Always (our default: 0.5) |
| **Weight Decay (L2)** | Penalizes large weights | Always (our default: 1e-4) |
| **Data Augmentation** | Random transforms on training images | Always |
| **Early Stopping** | Stop when validation stops improving | Always |
| **Batch Normalization** | Normalizes layer outputs | Built into ResNet |

---

## 5. AWS Services Explained

### Amazon S3 (Simple Storage Service)

**What:** Object storage (files in buckets).
**Analogy:** An infinite hard drive in the cloud.

Key features used:
- **Versioning:** Keep old versions of files (undo accidental deletes)
- **Lifecycle policies:** Automatically move old files to cheaper storage
- **Event notifications:** Trigger Lambda when a file is uploaded
- **Storage classes:** Standard ($0.023/GB), IA ($0.0125/GB), Glacier ($0.004/GB)

### Amazon SageMaker

**What:** Managed ML platform for training and deploying models.
**Analogy:** A data science workbench with built-in GPUs and deployment.

How we use it:
1. **Training jobs:** Upload code + data, SageMaker provisions GPU and trains
2. **Endpoints:** Host the model for real-time inference
3. **Auto-scaling:** Adjusts instances based on traffic
4. **Model Monitor:** Detects data drift in production

### AWS Lambda

**What:** Serverless compute — run code without managing servers.
**Analogy:** A vending machine. Put in input, get output. No maintenance.

Key properties:
- **Trigger-based:** Runs in response to events (API call, S3 upload, schedule)
- **Auto-scaling:** Handles 1 or 1000 concurrent requests automatically
- **Pay-per-use:** Billed per 1ms of execution time
- **Stateless:** Each invocation is independent (no shared memory)
- **Cold start:** First invocation takes 1-3 seconds (container initialization)

### API Gateway

**What:** Managed REST API service.
**Analogy:** A receptionist who routes requests, enforces rules, and protects the office.

Features:
- **Rate limiting:** Prevents abuse (100 req/sec limit)
- **Caching:** Stores responses to avoid duplicate Lambda calls
- **CORS:** Enables web browsers to call the API
- **API keys:** Track and limit usage per client
- **Stages:** dev/staging/prod environments from same API

### DynamoDB

**What:** Fully managed NoSQL database with millisecond latency.
**Analogy:** A super-fast phone book that auto-expands.

Key concepts:
- **Partition key:** Distributes data (like sharding). Choose a high-cardinality attribute (product_id, user_id)
- **Sort key:** Orders items within a partition (timestamp for chronological queries)
- **GSI (Global Secondary Index):** Query by different attributes (query by category)
- **TTL:** Auto-delete expired items (save money on old data)
- **On-demand billing:** Pay per read/write, no capacity planning

### Step Functions

**What:** Visual workflow orchestrator for multi-step processes.
**Analogy:** A flowchart that actually executes.

Our retraining pipeline:
```
PrepareData → TrainModel → EvaluateModel → [accuracy ≥ 90%?]
                                              ├─ Yes → DeployModel → NotifySuccess
                                              └─ No  → NotifyFailure
```

Built-in features:
- **Retries:** Automatically retry failed steps
- **Error handling:** Catch specific errors and route to recovery
- **Timeouts:** Prevent stuck executions
- **Visual debugging:** See exactly where a run failed in the console

### EventBridge

**What:** Serverless event bus for scheduling and routing events.
**We use it for:** Triggering weekly retraining with cron: `cron(0 2 ? * SUN *)`

### CloudWatch

**What:** Monitoring and observability platform.

Components:
- **Metrics:** Numerical data points (latency, error count, CPU usage)
- **Logs:** Text output from Lambda, SageMaker, etc.
- **Alarms:** Trigger notifications when metrics cross thresholds
- **Dashboards:** Visual display of system health

### IAM (Identity and Access Management)

**What:** Security and access control for AWS resources.
**Principle:** Least privilege — each service gets ONLY the permissions it needs.

Example: Lambda function gets:
- ✅ S3:PutObject (write images)
- ✅ SageMaker:InvokeEndpoint (make predictions)
- ✅ DynamoDB:PutItem (store results)
- ❌ S3:DeleteBucket (not needed!)
- ❌ SageMaker:DeleteEndpoint (not needed!)

---

## 6. MLOps & Production ML

### What is MLOps?

MLOps = ML + DevOps. Practices for deploying and maintaining ML systems in production.

**Key principles:**
1. **Automation:** Training pipeline runs without human intervention
2. **Reproducibility:** Any experiment can be exactly recreated
3. **Monitoring:** Detect model degradation before customers notice
4. **Version control:** Track changes to data, code, models, and configs
5. **Testing:** Validate model behavior before deployment

### Model Drift

Over time, the real world changes and the model's training data becomes stale.

**Types of drift:**
- **Data drift:** Input distribution changes (new product types, different image quality)
- **Concept drift:** The relationship between input and output changes (fashion trends)
- **Model decay:** Model performance degrades as the world changes

**Detection:** Compare incoming data distribution to training data distribution.
**Mitigation:** Automated retraining pipeline (our Step Functions workflow).

### A/B Testing for Models

Before fully deploying a new model, route a percentage of traffic to it:

```
Incoming Request
      │
  ┌───┴───┐
  │  90%  │  10%
  ▼       ▼
Model A  Model B
(current) (new)
```

Compare metrics (accuracy, latency, error rate). If Model B is better, gradually shift 100% of traffic.

SageMaker supports this with **production variants** on the same endpoint.

---

## 7. System Design Interview Questions

### "Design a product image classification system"

**Clarifying questions to ask:**
- How many categories? (5? 100? 10,000?)
- What's the expected QPS (queries per second)?
- What latency is acceptable? (real-time vs. batch?)
- How often does the model need to be updated?
- What's the budget?

**High-level architecture:**
1. API Gateway → Lambda (preprocess) → SageMaker (inference)
2. Results stored in DynamoDB for fast retrieval
3. Images stored in S3 for audit trail and retraining
4. Automated retraining pipeline with accuracy gates

**Scaling considerations:**
- SageMaker auto-scaling: 1-5 instances based on QPS
- Lambda concurrency: 1000 default, can increase
- DynamoDB on-demand: auto-scales read/write capacity
- API Gateway: rate limiting prevents abuse

### Common Follow-up Questions

**Q: How would you handle a new product category?**
A: Add labeled images for the new category, update num_classes, retrain, evaluate, deploy through the pipeline. The Step Functions workflow handles this automatically.

**Q: What if accuracy drops in production?**
A: CloudWatch alarms detect metric anomalies. SageMaker Model Monitor detects data drift. If triggered, the pipeline retrains on updated data. If retraining doesn't help, fall back to the previous model version (saved in S3 with versioning).

**Q: How would you handle 100x more traffic?**
A: SageMaker auto-scaling handles inference. Add API Gateway caching for repeated images. For batch processing, use SageMaker Batch Transform. For edge cases, consider multi-model endpoints to share instance resources.

**Q: How would you reduce costs further?**
A: Spot instances for training (70% savings). SageMaker Serverless Inference for low-traffic periods. S3 Intelligent-Tiering for storage. Reserved capacity for steady-state endpoints.

---

## Quick Reference: Key Numbers to Know

| Metric | Value | Why It Matters |
|--------|-------|----------------|
| ResNet18 parameters | 11.7M | Shows model complexity |
| ImageNet classes | 1,000 | Source of pre-trained features |
| Our classes | 5 | Target task complexity |
| Input resolution | 224×224 | Standard for ResNet |
| Training batch size | 32 | Balance of speed and stability |
| Learning rate | 0.001 | Adam default, works well |
| Training time | ~35 min | On ml.g4dn.xlarge |
| Inference latency (p99) | < 800ms | On ml.m5.large (CPU) |
| Monthly cost | ~$128 | With optimizations |
| Cold start (Lambda) | ~1-3 sec | First invocation only |
