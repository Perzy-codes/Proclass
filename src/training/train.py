"""
Main Training Script — SageMaker Entry Point
==============================================

This script is the entry point for SageMaker training jobs. SageMaker runs it
inside a Docker container with our code, data, and a GPU.

CONCEPT: How SageMaker Training Works
---------------------------------------
1. You call estimator.fit({'train': 's3://bucket/train/', 'val': 's3://bucket/val/'})
2. SageMaker provisions an EC2 instance (e.g., ml.g4dn.xlarge with NVIDIA T4 GPU)
3. SageMaker downloads training data from S3 to the instance
4. SageMaker runs this script (train.py) with environment variables:
     SM_CHANNEL_TRAIN  = /opt/ml/input/data/train/    (training data path)
     SM_CHANNEL_VAL    = /opt/ml/input/data/val/       (validation data path)
     SM_MODEL_DIR      = /opt/ml/model/                (where to save the model)
     SM_OUTPUT_DIR     = /opt/ml/output/               (for logs/artifacts)
     SM_NUM_GPUS       = 1
5. After training, SageMaker uploads SM_MODEL_DIR to S3 as model.tar.gz
6. The instance is automatically terminated (you only pay for training time!)

CONCEPT: SageMaker Environment Variables
------------------------------------------
SageMaker uses a convention where:
  SM_CHANNEL_{NAME}  = path to data channel (train, val, test)
  SM_MODEL_DIR       = directory where model artifacts must be saved
  SM_HP_{NAME}       = hyperparameters passed from estimator config
  SM_NUM_GPUS        = number of GPUs available

These are set automatically — you just read them via argparse or os.environ.

CONCEPT: Training Loop Anatomy
-------------------------------
Every training loop follows the same pattern:

  for epoch in range(num_epochs):           # Repeat over entire dataset
      model.train()                          # Enable dropout, batch norm training mode
      for batch in train_loader:             # Iterate over mini-batches
          optimizer.zero_grad()              # Clear old gradients
          outputs = model(inputs)            # Forward pass
          loss = criterion(outputs, labels)  # Compute loss
          loss.backward()                    # Backpropagation (compute gradients)
          optimizer.step()                   # Update weights
      
      model.eval()                           # Disable dropout for validation
      validate(model, val_loader)            # Check performance on held-out data
      scheduler.step()                       # Adjust learning rate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Tuple

# Local imports
from model import ProductClassifier, get_model
from dataset import create_data_loaders, ProductImageDataset, get_val_transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch and return average loss and accuracy.
    
    CONCEPT: One Epoch
    -------------------
    An epoch = one complete pass through the entire training dataset.
    
    With 8,000 training images and batch_size=32:
      - Each epoch has 8000/32 = 250 iterations (batches)
      - Each iteration processes 32 images
      - After 250 iterations, every image has been seen once → 1 epoch
    
    Typical training uses 10-50 epochs. More epochs = more chances to learn,
    but too many → overfitting (memorizing training data).
    """
    model.train()  # IMPORTANT: enables dropout and batch norm training mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to GPU (if available)
        # CONCEPT: .to(device)
        # Tensors must be on the same device as the model.
        # If model is on GPU, data must also be on GPU.
        images = images.to(device)
        labels = labels.to(device)
        
        # ============================================================
        # THE CORE TRAINING STEP (5 lines that power all of deep learning)
        # ============================================================
        
        # 1. Zero gradients from previous iteration
        # CONCEPT: Why zero_grad()?
        # PyTorch ACCUMULATES gradients by default (adds new to old).
        # This is useful for gradient accumulation across micro-batches,
        # but normally we want fresh gradients each iteration.
        optimizer.zero_grad()
        
        # 2. Forward pass: images → model → predictions
        outputs = model(images)  # Shape: (batch_size, 5) — logits for each class
        
        # 3. Compute loss: how wrong are our predictions?
        # CONCEPT: CrossEntropyLoss
        # Combines log_softmax + NLLLoss. It measures how far our predicted
        # probability distribution is from the true distribution.
        # 
        # If true label = "electronics" (index 0):
        #   Good prediction: [0.9, 0.02, 0.03, 0.02, 0.03] → low loss
        #   Bad prediction:  [0.1, 0.3, 0.2, 0.3, 0.1]    → high loss
        #
        # The loss is always a SINGLE NUMBER — we need one number to optimize.
        loss = criterion(outputs, labels)
        
        # 4. Backward pass: compute gradients
        # CONCEPT: Backpropagation
        # Starting from the loss, PyTorch computes ∂loss/∂weight for every
        # trainable parameter. This tells us "how should each weight change
        # to reduce the loss?"
        #
        # These gradients are stored in param.grad for each parameter.
        loss.backward()
        
        # 5. Update weights using gradients
        # CONCEPT: Optimizer Step
        # Adam updates each weight: w_new = w_old - lr * (adjusted_gradient)
        # Adam is "smarter" than plain SGD — it adapts the learning rate
        # per-parameter based on gradient history (momentum + RMSprop).
        optimizer.step()
        
        # ============================================================
        # END OF CORE TRAINING STEP
        # ============================================================
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get predicted class (highest logit)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Log progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}: "
                f"Loss={loss.item():.4f}"
            )
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on validation set.
    
    CONCEPT: Validation
    --------------------
    Validation measures how well the model generalizes to UNSEEN data.
    
    Training accuracy tells you: "How well did the model memorize training data?"
    Validation accuracy tells you: "How well will the model work on new data?"
    
    If training accuracy >> validation accuracy → OVERFITTING
    The model memorized training data but can't generalize.
    
    Solutions:
    - More data / more augmentation
    - More dropout / weight decay (regularization)
    - Earlier stopping (stop training when val accuracy peaks)
    - Simpler model (fewer parameters)
    """
    model.eval()  # IMPORTANT: disables dropout, uses running stats for batch norm
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # CONCEPT: torch.no_grad()
    # During validation, we don't need gradients.
    # no_grad() saves memory and speeds up computation.
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def save_model(model: nn.Module, model_dir: str, metrics: Dict) -> None:
    """
    Save model artifacts for SageMaker deployment.
    
    CONCEPT: What Gets Saved
    -------------------------
    We save three things:
    1. model.pth — The model weights (state_dict)
    2. model_info.json — Metadata (categories, accuracy, config)
    3. The training script files — Needed for inference
    
    CONCEPT: state_dict vs. Full Model
    ------------------------------------
    torch.save(model.state_dict(), path)  — saves ONLY the weights
    torch.save(model, path)               — saves weights + architecture
    
    We use state_dict because:
    - Smaller file size
    - More portable (doesn't depend on exact code structure)
    - Recommended by PyTorch documentation
    - Easier to load with modified architecture
    """
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model weights saved to {model_path}")
    
    # Save metadata for inference
    info = {
        'categories': ProductClassifier.CATEGORIES,
        'num_classes': model.num_classes,
        'input_size': list(ProductClassifier.INPUT_SIZE),
        'imagenet_mean': ProductClassifier.IMAGENET_MEAN,
        'imagenet_std': ProductClassifier.IMAGENET_STD,
        'metrics': metrics,
        'total_params': model.get_total_params(),
        'trainable_params': model.get_trainable_params(),
    }
    
    info_path = os.path.join(model_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    logger.info(f"Model info saved to {info_path}")


def train(args) -> Dict:
    """
    Main training function.
    
    CONCEPT: Training Recipe
    -------------------------
    1. Setup: device, model, loss function, optimizer, scheduler
    2. Training loop: for each epoch, train → validate → adjust LR
    3. Early stopping: stop if validation doesn't improve
    4. Save: best model weights + metadata
    """
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    # Detect GPU
    # CONCEPT: CUDA
    # CUDA is NVIDIA's GPU computing platform. PyTorch uses it to run
    # tensor operations on GPU (10-100x faster than CPU for training).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # Create model
    model_config = {
        'num_classes': args.num_classes,
        'pretrained': True,
        'freeze_layers': args.freeze_layers,
        'dropout_rate': args.dropout_rate
    }
    model = get_model(model_config)
    model.to(device)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=224
    )
    
    # Loss function
    # CONCEPT: CrossEntropyLoss
    # The standard loss for multi-class classification.
    # It measures the "distance" between predicted probabilities and true labels.
    # Internally: log_softmax(predictions) then negative log likelihood
    # 
    # Optional: Pass class weights to handle imbalanced datasets
    # criterion = nn.CrossEntropyLoss(weight=dataset.get_class_weights().to(device))
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    # CONCEPT: Adam Optimizer
    # Adam (Adaptive Moment Estimation) maintains per-parameter learning rates.
    # It combines:
    #   - Momentum: Uses exponential moving average of gradients (smooths noise)
    #   - RMSprop: Adapts learning rate based on gradient magnitude
    #
    # Only optimize TRAINABLE parameters (not frozen ones)
    # filter(lambda p: p.requires_grad, ...) skips frozen parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay  # L2 regularization
        # CONCEPT: Weight Decay (L2 Regularization)
        # Adds a penalty for large weights: loss += weight_decay * ||weights||^2
        # This discourages the model from relying too heavily on any single feature.
        # Typical value: 1e-4 to 1e-5
    )
    
    # Learning rate scheduler
    # CONCEPT: Cosine Annealing
    # Gradually decreases learning rate following a cosine curve:
    #   LR starts high → slowly decreases → reaches minimum at end
    #
    # Why? High LR early on explores the loss landscape broadly.
    # Low LR later on fine-tunes to a precise minimum.
    #
    #   LR │ ╲
    #      │   ╲
    #      │    ╲
    #      │     ╲___
    #      └──────────── Epoch
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01  # Minimum LR = 1% of initial
    )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Trainable parameters: {model.get_trainable_params():,}")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )
        
        # Save best model
        # CONCEPT: Early Stopping
        # If validation accuracy doesn't improve for `patience` epochs, stop.
        # This prevents overfitting — the model starts to memorize training data
        # instead of learning general patterns.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            save_model(model, args.model_dir, {
                'best_val_accuracy': best_val_acc,
                'best_epoch': best_epoch,
                'final_train_accuracy': train_acc,
                'final_train_loss': train_loss,
            })
            logger.info(f"  ★ New best model! Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1}. "
                    f"Best val accuracy: {best_val_acc:.2f}% at epoch {best_epoch}"
                )
                break
    
    # Save training history
    history_path = os.path.join(args.model_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\nTraining complete!")
    logger.info(f"  Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    
    return {
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'history': history
    }


if __name__ == '__main__':
    # CONCEPT: argparse + SageMaker
    # SageMaker passes hyperparameters as command-line arguments.
    # It also sets environment variables for data/model directories.
    # argparse reads both.
    
    parser = argparse.ArgumentParser(description='Product Classifier Training')
    
    # Hyperparameters (passed from SageMaker estimator config)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout-rate', type=float, default=0.5)
    parser.add_argument('--freeze-layers', type=int, default=6)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # SageMaker environment variables (set automatically)
    parser.add_argument('--train-dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', './data/train'))
    parser.add_argument('--val-dir', type=str,
                        default=os.environ.get('SM_CHANNEL_VALIDATION', './data/val'))
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', './model'))
    
    args = parser.parse_args()
    
    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Run training
    train(args)
