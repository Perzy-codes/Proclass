"""
Product Classifier CNN Architecture
====================================

This module defines the neural network used to classify product images.

CONCEPT: Transfer Learning
--------------------------
Instead of training a CNN from scratch (which needs millions of images), we take a model
that was already trained on ImageNet (1.2 million images, 1000 classes) and adapt it.

Think of it like this: A photographer who's spent years learning to see edges, colors,
textures, and shapes can quickly learn to identify new objects. They don't need to 
re-learn "what is an edge?" — they already know that. They just need to learn 
"what does a book look like vs. a laptop?"

That's exactly what transfer learning does:
  1. Take ResNet18 (pre-trained on ImageNet) — it already knows how to "see"
  2. Freeze the early layers (edges, textures) — these are universal
  3. Fine-tune later layers (object parts, high-level features) — these are task-specific
  4. Replace the final classification layer (1000 → 5 categories)

CONCEPT: ResNet (Residual Networks)
------------------------------------
ResNet solved the "vanishing gradient problem" — in very deep networks, gradients
become tiny during backpropagation, making early layers nearly impossible to train.

ResNet's solution: SKIP CONNECTIONS (aka residual connections)

    Normal path:     Input → Conv → BatchNorm → ReLU → Conv → BatchNorm → Output
    Skip connection: Input ──────────────────────────────────────────────→ + → ReLU → Output

    Output = F(x) + x     (instead of just F(x))

Why this works: If a layer can't learn anything useful, F(x) ≈ 0, so Output ≈ x.
The network can "skip" layers that aren't helpful. This makes it easy to train
very deep networks (18, 34, 50, 101, 152 layers).

ResNet18 = 18 weight layers = 8 residual blocks + initial conv + final FC

CONCEPT: Why ResNet18 over ResNet50?
-------------------------------------
For 5 classes with ~10K training images:
  - ResNet18: 11.7M params → faster training, faster inference, less overfitting risk
  - ResNet50: 25.6M params → overkill for 5 classes, slower, more expensive
  - Accuracy difference: < 1% for this task
  - Inference speed: ResNet18 is 2x faster → lower latency, cheaper endpoint
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONCEPT: What Each Layer "Sees"
# ============================================================================
# Layer 1 (64 filters):  Edges, gradients, color blobs
# Layer 2 (128 filters): Textures, patterns, simple shapes
# Layer 3 (256 filters): Object parts (buttons, screens, fabric patterns)
# Layer 4 (512 filters): Object-level features (whole keyboard, clothing style)
# FC Layer (512 → 5):    Final category decision
#
# We freeze Layers 1-2 (universal features) and fine-tune Layers 3-4 + FC
# (task-specific features). This is called "partial fine-tuning."
# ============================================================================


class ProductClassifier(nn.Module):
    """
    Product image classifier using ResNet18 backbone with transfer learning.
    
    Classifies images into: electronics, clothing, furniture, books, toys
    
    Architecture choices explained:
    - ResNet18 backbone: Good accuracy/speed tradeoff for 5 classes
    - Dropout (0.5): Prevents overfitting on small dataset
    - Frozen early layers: Preserves universal features, reduces training time
    
    CONCEPT: nn.Module
    ------------------
    Every PyTorch model inherits from nn.Module. This gives us:
    - Automatic parameter tracking (model.parameters())
    - GPU/CPU movement (model.to(device))
    - Save/load functionality (torch.save/load)
    - Train/eval mode switching (model.train()/model.eval())
    """
    
    # Class-level constants
    CATEGORIES = ['apparel', 'footwear', 'accessories', 'bottomwear', 'personal_care']
    INPUT_SIZE = (224, 224)  # ResNet expects 224x224 images
    
    # CONCEPT: ImageNet normalization values
    # These are the mean and std of the ImageNet dataset (RGB channels).
    # Since ResNet was trained on ImageNet with these values, we MUST use the
    # same normalization on our images — otherwise the pre-trained weights
    # won't work correctly (the features would be scaled differently).
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_layers: int = 6,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the product classifier.
        
        Args:
            num_classes: Number of product categories (default: 5)
            pretrained: Use ImageNet pre-trained weights (default: True)
            freeze_layers: Number of layers to freeze from the start.
                          Higher = fewer trainable params = faster training
                          but less flexibility. 6 freezes through Layer2.
            dropout_rate: Dropout probability before final FC layer.
                         0.5 = randomly zero out 50% of neurons during training.
                         
        CONCEPT: Dropout
        ----------------
        During training, dropout randomly sets neurons to 0 with probability p.
        This forces the network to not rely on any single neuron — it must learn
        redundant representations. This prevents overfitting (memorizing training data).
        
        During evaluation (model.eval()), dropout is automatically disabled —
        all neurons participate, and outputs are scaled by (1-p) to compensate.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pre-trained ResNet18
        # CONCEPT: weights parameter
        # 'IMAGENET1K_V1' = pre-trained on ImageNet with standard training
        # None = random initialization (training from scratch)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Freeze early layers (universal features that don't need re-learning)
        self._freeze_early_layers(freeze_layers)
        
        # Get the feature dimension from the original FC layer
        # ResNet18's FC layer: Linear(512, 1000) — 512 features, 1000 ImageNet classes
        in_features = self.backbone.fc.in_features  # 512
        
        # Replace the final classification head
        # CONCEPT: nn.Sequential
        # Chains multiple layers together. Input flows through each in order.
        # This is cleaner than writing separate forward() logic for each layer.
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Regularization
            nn.Linear(in_features, 256),  # Bottleneck: 512 → 256
            nn.ReLU(inplace=True),        # Non-linearity
            nn.Dropout(p=dropout_rate / 2),  # Lighter dropout
            nn.Linear(256, num_classes)   # Final: 256 → 5 categories
        )
        
        # CONCEPT: Why a 2-layer head instead of just Linear(512, 5)?
        # The intermediate layer (256 neurons) acts as a "feature adapter" —
        # it learns to combine ResNet's 512 ImageNet features into 256 features
        # that are specifically useful for product classification.
        # This is called a "bottleneck" and it improves accuracy by ~1-2%.
        
        self._log_model_info()
    
    def _freeze_early_layers(self, num_layers: int) -> None:
        """
        Freeze the first `num_layers` parameter groups.
        
        CONCEPT: Freezing Layers
        ------------------------
        param.requires_grad = False means:
          - This parameter will NOT be updated during training
          - Gradients won't be computed for it (saves memory + compute)
          - The pre-trained values are preserved exactly
        
        Why freeze?
          1. Early layers learn universal features (edges, textures) — reusable
          2. Reduces trainable parameters → less overfitting risk
          3. Faster training (fewer gradients to compute)
          4. Prevents "catastrophic forgetting" (overwriting good features)
        """
        all_params = list(self.backbone.parameters())
        frozen_count = 0
        
        for i, param in enumerate(all_params):
            if i < num_layers:
                param.requires_grad = False
                frozen_count += param.numel()
        
        total_params = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            f"Frozen {frozen_count:,} / {total_params:,} parameters "
            f"({frozen_count/total_params*100:.1f}%)"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image tensor → class logits.
        
        CONCEPT: Forward Pass
        ---------------------
        This is the function that runs when you call model(images).
        
        Input:  x with shape (batch_size, 3, 224, 224)
                - batch_size: number of images processed together
                - 3: RGB color channels
                - 224, 224: image height and width
        
        Output: logits with shape (batch_size, 5)
                - Raw scores for each category (not probabilities yet!)
                - To get probabilities, apply softmax: F.softmax(output, dim=1)
        
        CONCEPT: Logits vs Probabilities
        ---------------------------------
        Logits are raw model outputs (can be any real number).
        Probabilities are logits after softmax (between 0 and 1, sum to 1).
        
        We output logits (not probabilities) because:
        - CrossEntropyLoss expects logits (it applies softmax internally)
        - Applying softmax twice would give wrong gradients
        - We only need probabilities at inference time
        """
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict category and confidence for input images.
        
        Returns:
            Tuple of (predicted_classes, confidence_scores)
            
        CONCEPT: torch.no_grad()
        ------------------------
        During inference, we don't need to compute gradients (no training).
        torch.no_grad() tells PyTorch to skip gradient computation:
          - Saves memory (no gradient tensors stored)
          - Faster execution (no backward pass preparation)
          - Required for production inference
        """
        self.eval()  # Switch to evaluation mode (disables dropout, batch norm uses running stats)
        
        with torch.no_grad():
            logits = self.forward(x)
            
            # CONCEPT: softmax converts logits to probabilities
            # dim=1 means "apply softmax across classes" (not across batch)
            # Example: logits [2.1, 0.5, -1.0, 0.3, -0.8]
            #       → probs  [0.55, 0.11, 0.02, 0.09, 0.03]  (sum ≈ 1.0)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Get the highest probability and its index
            confidence, predicted = probabilities.max(dim=1)
        
        return predicted, confidence
    
    def predict_with_all_scores(self, x: torch.Tensor) -> Dict:
        """
        Get predictions with full probability distribution.
        
        Useful for:
        - Debugging: See if the model is confused between categories
        - Thresholding: Only accept predictions above a confidence threshold
        - Logging: Track prediction distributions over time
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        results = []
        for probs in probabilities:
            scores = {
                cat: float(prob) 
                for cat, prob in zip(self.CATEGORIES, probs)
            }
            # Sort by confidence (highest first)
            sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
            results.append(sorted_scores)
        
        return results
    
    def get_trainable_params(self) -> int:
        """Count parameters that will be updated during training."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count all parameters (frozen + trainable)."""
        return sum(p.numel() for p in self.parameters())
    
    def _log_model_info(self) -> None:
        """Log model architecture summary."""
        total = self.get_total_params()
        trainable = self.get_trainable_params()
        frozen = total - trainable
        
        logger.info(f"Model: ProductClassifier (ResNet18 backbone)")
        logger.info(f"  Total parameters:     {total:>12,}")
        logger.info(f"  Trainable parameters: {trainable:>12,}")
        logger.info(f"  Frozen parameters:    {frozen:>12,}")
        logger.info(f"  Categories: {self.CATEGORIES}")
        logger.info(f"  Dropout rate: {self.dropout_rate}")


def get_model(config: Optional[Dict] = None) -> ProductClassifier:
    """
    Factory function to create a model with config.
    
    CONCEPT: Factory Pattern
    ------------------------
    Instead of calling ProductClassifier() directly, we use a factory
    function that reads configuration. This lets us change model settings
    (num_classes, dropout, etc.) without changing code — just update config.
    
    This pattern is common in ML codebases because:
    - Training and inference may use different configs
    - Easy to experiment with hyperparameters
    - Clean separation of config from code
    """
    if config is None:
        config = {}
    
    return ProductClassifier(
        num_classes=config.get('num_classes', 5),
        pretrained=config.get('pretrained', True),
        freeze_layers=config.get('freeze_layers', 6),
        dropout_rate=config.get('dropout_rate', 0.5)
    )


# ============================================================================
# INTERVIEW TALKING POINTS
# ============================================================================
# Q: Why ResNet18 and not a transformer (like ViT)?
# A: For 5 classes with ~10K images, ResNet18 is sufficient and much cheaper
#    to host. ViT needs more data and more compute. Engineering is about
#    choosing the RIGHT tool, not the newest tool.
#
# Q: How would you improve accuracy?
# A: 1. More data (data augmentation, synthetic data)
#    2. Better augmentation (CutMix, MixUp, AutoAugment)
#    3. Learning rate scheduling (cosine annealing)
#    4. Ensemble of 2-3 models
#    5. Test-time augmentation (TTA)
#
# Q: What if you needed 100 categories instead of 5?
# A: Switch to ResNet50 or EfficientNet-B3, add more training data,
#    use label smoothing to handle similar categories, and consider
#    hierarchical classification.
#
# Q: How do you handle class imbalance?
# A: 1. Weighted loss (give rare classes higher weight)
#    2. Oversampling minority classes
#    3. Focal loss (down-weights easy examples)
#    4. Data augmentation on minority classes
# ============================================================================
