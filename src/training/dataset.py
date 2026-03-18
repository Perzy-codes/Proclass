"""
Custom Dataset with Data Augmentation
======================================

This module handles loading product images from S3 and applying data augmentation
to improve model training.

CONCEPT: Data Augmentation
---------------------------
Data augmentation creates "new" training images by randomly transforming existing ones.
This teaches the model to be invariant to these transformations.

Example: A photo of a laptop is still a laptop whether it's:
  - Flipped horizontally (RandomHorizontalFlip)
  - Slightly rotated (RandomRotation)
  - Cropped differently (RandomResizedCrop)
  - Brighter or darker (ColorJitter)

Why this matters:
  - Effectively multiplies dataset size (10K images → behaves like 50K+)
  - Reduces overfitting (model sees different versions each epoch)
  - Makes model robust to real-world variations (lighting, angle, etc.)
  - FREE improvement — no need to collect more data

CONCEPT: Train vs. Validation Transforms
-----------------------------------------
Training: Apply augmentation (random flips, rotations, color changes)
Validation/Test: NO augmentation — we need consistent, reproducible results

Both need: Resize → ToTensor → Normalize (with ImageNet stats)

CONCEPT: Why Normalize with ImageNet Stats?
--------------------------------------------
The pre-trained ResNet was trained on ImageNet with specific mean/std normalization.
If we normalize our images differently, the pre-trained features won't work correctly —
it's like feeding the model input in a different "language."

  mean = [0.485, 0.456, 0.406]   ← ImageNet RGB channel means
  std  = [0.229, 0.224, 0.225]   ← ImageNet RGB channel std devs
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import logging
from typing import Tuple, Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# CONCEPT: torchvision.transforms
# ============================================================================
# Transforms are composable image transformations. They form a pipeline:
#   Input Image (PIL) → Transform1 → Transform2 → ... → Tensor
#
# Key transforms explained:
#   Resize(256): Scale shorter edge to 256 pixels
#   CenterCrop(224): Take 224×224 from center (deterministic)
#   RandomResizedCrop(224): Random crop + resize (random — augmentation!)
#   RandomHorizontalFlip(p=0.5): 50% chance to mirror horizontally
#   ColorJitter: Randomly adjust brightness, contrast, saturation, hue
#   ToTensor(): PIL Image (0-255) → Tensor (0.0-1.0), and HWC → CHW format
#   Normalize(): Subtract mean, divide by std (standardize to ImageNet scale)
# ============================================================================


def get_train_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Training transforms with data augmentation.
    
    CONCEPT: Compose
    -----------------
    transforms.Compose([...]) chains transforms sequentially.
    Each transform receives the output of the previous one.
    
    The order matters:
      1. Spatial transforms first (crop, flip, rotate) — work on PIL images
      2. ToTensor() — converts to tensor (required before Normalize)
      3. Normalize() — must be last (works on tensors)
    """
    return transforms.Compose([
        # RandomResizedCrop: Randomly crop a portion (8%-100%) of the image,
        # then resize to 224×224. This teaches the model to recognize objects
        # at different scales and positions.
        transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
        
        # 50% chance to flip horizontally. A laptop is still a laptop when mirrored.
        # NOTE: We don't flip vertically because upside-down products are rare.
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Randomly change brightness, contrast, saturation, and hue.
        # Makes the model robust to different lighting conditions.
        # Values: brightness=±20%, contrast=±20%, saturation=±20%, hue=±10%
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        
        # Small random rotation (±15 degrees).
        # Products in photos aren't always perfectly aligned.
        transforms.RandomRotation(degrees=15),
        
        # Convert PIL Image (H, W, C) with values [0, 255]
        # to Tensor (C, H, W) with values [0.0, 1.0]
        # CONCEPT: PyTorch uses Channel-First format (C, H, W)
        # while most image libraries use Channel-Last (H, W, C)
        transforms.ToTensor(),
        
        # Normalize using ImageNet statistics (REQUIRED for transfer learning)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Validation/test transforms — NO augmentation.
    
    We resize to 256 then center-crop to 224 (standard practice).
    This gives a small border that avoids edge artifacts.
    
    IMPORTANT: Validation transforms must be deterministic!
    We need the same results every time for fair evaluation.
    """
    return transforms.Compose([
        transforms.Resize(256),           # Resize shorter edge to 256
        transforms.CenterCrop(input_size), # Take 224×224 from center
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class ProductImageDataset(Dataset):
    """
    Custom dataset for product images organized in folders by category.
    
    CONCEPT: PyTorch Dataset
    -------------------------
    A Dataset must implement:
      __len__(): Return the total number of samples
      __getitem__(idx): Return one sample (image, label) at index idx
    
    PyTorch's DataLoader then uses these to:
      - Load batches of data
      - Shuffle samples (for training)
      - Use multiple workers for parallel loading
      - Pin memory for faster GPU transfer
    
    Expected directory structure:
        data_dir/
        ├── electronics/
        │   ├── img001.jpg
        │   └── img002.jpg
        ├── clothing/
        │   ├── img003.jpg
        │   └── img004.jpg
        └── ...
    """
    
    CATEGORIES = ['apparel', 'footwear', 'accessories', 'bottomwear', 'personal_care']
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        max_samples_per_class: Optional[int] = None
    ):
        """
        Args:
            data_dir: Root directory containing category folders
            transform: Image transforms to apply
            max_samples_per_class: Limit samples per category (for quick testing)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []  # (image_path, label_index)
        
        # Build label mapping: category name → integer index
        # CONCEPT: Neural networks output numbers, not strings.
        # We map: 'electronics' → 0, 'clothing' → 1, etc.
        # The model outputs a vector of 5 probabilities, where index 0
        # corresponds to 'electronics', index 1 to 'clothing', etc.
        self.class_to_idx = {cat: idx for idx, cat in enumerate(self.CATEGORIES)}
        self.idx_to_class = {idx: cat for cat, idx in self.class_to_idx.items()}
        
        # Scan directories and build sample list
        self._load_samples(max_samples_per_class)
        
        logger.info(f"Loaded {len(self.samples)} images from {data_dir}")
        self._log_class_distribution()
    
    def _load_samples(self, max_per_class: Optional[int]) -> None:
        """Scan directory structure and register all valid image files."""
        for category in self.CATEGORIES:
            category_dir = self.data_dir / category
            if not category_dir.exists():
                logger.warning(f"Category directory not found: {category_dir}")
                continue
            
            label = self.class_to_idx[category]
            count = 0
            
            for img_path in sorted(category_dir.iterdir()):
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((str(img_path), label))
                    count += 1
                    
                    if max_per_class and count >= max_per_class:
                        break
    
    def _log_class_distribution(self) -> None:
        """Log how many images per category (important for detecting imbalance)."""
        distribution = {}
        for _, label in self.samples:
            cat = self.idx_to_class[label]
            distribution[cat] = distribution.get(cat, 0) + 1
        
        logger.info("Class distribution:")
        for cat, count in distribution.items():
            logger.info(f"  {cat}: {count} images")
        
        # Warn about class imbalance
        # CONCEPT: Class Imbalance
        # If one category has 5000 images and another has 500, the model
        # will be biased toward the majority class. Solutions:
        # 1. Weighted loss (give minority classes higher loss weight)
        # 2. Oversampling (duplicate minority samples)
        # 3. Data augmentation (more augmentation for minority classes)
        if distribution:
            max_count = max(distribution.values())
            min_count = min(distribution.values())
            if max_count > 3 * min_count:
                logger.warning(
                    f"Class imbalance detected! Max: {max_count}, Min: {min_count}. "
                    f"Consider using weighted loss or oversampling."
                )
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return one sample.
        
        CONCEPT: Lazy Loading
        ----------------------
        We don't load all images into memory at initialization.
        Instead, we load each image on-demand when __getitem__ is called.
        
        This is crucial because:
        - 10,000 images at 224×224×3 = ~1.4 GB in memory
        - With augmentation, each image is different every time
        - DataLoader handles batching and parallel loading
        
        CONCEPT: Error Handling in Datasets
        ------------------------------------
        Real-world data is messy. Images might be corrupted, truncated,
        or in unexpected formats. We handle errors gracefully by returning
        a blank tensor instead of crashing the entire training run.
        """
        img_path, label = self.samples[idx]
        
        try:
            # Load image and convert to RGB
            # CONCEPT: Why .convert('RGB')?
            # Some images might be grayscale (1 channel), RGBA (4 channels),
            # or in other formats. ResNet expects exactly 3 channels (RGB).
            # .convert('RGB') standardizes everything.
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image tensor as fallback
            # This prevents one bad image from crashing training
            if self.transform:
                blank = Image.new('RGB', (224, 224), color=(128, 128, 128))
                return self.transform(blank), label
            return torch.zeros(3, 224, 224), label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for handling imbalance.
        
        CONCEPT: Weighted Loss
        -----------------------
        If we have:
          electronics: 3000 images
          books: 500 images
        
        Without weights: The model learns electronics much better (more examples)
        With weights: books gets 6x higher loss → model pays more attention to books
        
        Formula: weight[i] = total_samples / (num_classes * count[i])
        
        This ensures each class contributes equally to the total loss,
        regardless of how many samples it has.
        """
        class_counts = [0] * self.num_classes
        for _, label in self.samples:
            class_counts[label] += 1
        
        total = sum(class_counts)
        weights = [
            total / (self.num_classes * count) if count > 0 else 0.0
            for count in class_counts
        ]
        
        return torch.FloatTensor(weights)
    
    @property
    def num_classes(self) -> int:
        return len(self.CATEGORIES)


def create_data_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 224
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    CONCEPT: DataLoader
    --------------------
    DataLoader wraps a Dataset and provides:
      - Batching: Groups samples into batches (batch_size=32 → 32 images at once)
      - Shuffling: Randomizes order each epoch (training only!)
      - Parallel loading: Uses multiple processes (num_workers=4)
      - Memory pinning: Pre-copies data to GPU memory for faster transfer
    
    CONCEPT: Batch Size
    --------------------
    batch_size = number of images processed together in one forward pass.
    
    Too small (1-4): Noisy gradients, slow training, poor GPU utilization
    Too large (256+): Smooths out useful gradient noise, may need learning rate warmup
    Sweet spot (16-64): Good balance for most tasks. 32 is a safe default.
    
    CONCEPT: num_workers
    ---------------------
    Number of parallel processes for data loading.
    Rule of thumb: num_workers = 4 * num_GPUs
    Too few: GPU waits for data (data bottleneck)
    Too many: Context switching overhead
    
    CONCEPT: pin_memory
    --------------------
    When True, DataLoader copies tensors into CUDA pinned (page-locked) memory
    before transferring to GPU. This makes the GPU transfer asynchronous and faster.
    Only set True when using GPU training.
    """
    train_dataset = ProductImageDataset(
        data_dir=train_dir,
        transform=get_train_transforms(input_size)
    )
    
    val_dataset = ProductImageDataset(
        data_dir=val_dir,
        transform=get_val_transforms(input_size)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Randomize order each epoch (training only!)
        num_workers=num_workers,
        pin_memory=True,       # Faster GPU transfer
        drop_last=True         # Drop incomplete last batch (avoids BatchNorm issues)
        # CONCEPT: drop_last=True
        # If we have 1000 images and batch_size=32, the last batch has 8 images.
        # BatchNorm with 8 samples gives unreliable statistics.
        # drop_last=True discards this small batch. We lose a few samples
        # but get more stable training.
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # No shuffling for validation (reproducibility!)
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False        # Evaluate ALL validation samples
    )
    
    return train_loader, val_loader
