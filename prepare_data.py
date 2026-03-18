"""
Data Preparation Script — Kaggle Fashion Product Images → Project Format
=========================================================================

Downloads and organizes the Kaggle "Fashion Product Images (Small)" dataset
into the folder structure expected by our training pipeline.

Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
Size: ~44,000 real e-commerce product images, ~600 MB
License: CC0 Public Domain

USAGE:
  1. Download dataset from Kaggle (requires free account):
     kaggle datasets download -d paramaggarwal/fashion-product-images-small
     
  2. Unzip to get: styles.csv + images/ folder
  
  3. Run this script:
     python prepare_data.py --source /path/to/kaggle/download --output ./data
     
This will create:
  data/
  ├── train/
  │   ├── apparel/      (~11,795 images)
  │   ├── footwear/     (~7,377 images)
  │   ├── accessories/  (~7,458 images)
  │   ├── bottomwear/   (~2,855 images)
  │   └── personal_care/ (~1,804 images)
  ├── val/
  │   ├── apparel/      (~1,474 images)
  │   └── ...
  └── test/
      ├── apparel/      (~1,474 images)
      └── ...

Total: ~39,114 real product images across 5 categories
"""

import csv
import os
import shutil
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# ============================================================================
# CATEGORY MAPPING
# ============================================================================
# Maps Kaggle's 143 articleTypes into our 5 product categories.
# 
# Design decisions:
# - 5 categories keeps the model focused and achievable with transfer learning
# - Each category is VISUALLY distinct (important for CNN features)
# - Smallest class has 2,256 images (enough for transfer learning)
# - Total: 39,114 images (88% of the full dataset)
# ============================================================================

ARTICLE_TO_CATEGORY: Dict[str, str] = {}

# Category 1: Apparel (upper body clothing — shirts, tops, kurtas, jackets)
for t in ['Tshirts', 'Shirts', 'Tops', 'Kurtas', 'Sweatshirts', 'Jackets',
          'Sweaters', 'Blazers', 'Nehru Jackets', 'Rain Jacket']:
    ARTICLE_TO_CATEGORY[t] = 'apparel'

# Category 2: Footwear (all shoe types — casual, sports, formal, sandals)
for t in ['Casual Shoes', 'Sports Shoes', 'Heels', 'Flip Flops', 'Sandals',
          'Formal Shoes', 'Flats', 'Sports Sandals']:
    ARTICLE_TO_CATEGORY[t] = 'footwear'

# Category 3: Accessories (watches, bags, wallets, belts, jewelry, eyewear)
for t in ['Watches', 'Handbags', 'Wallets', 'Backpacks', 'Belts', 'Sunglasses',
          'Earrings', 'Clutches', 'Jewellery Set', 'Pendant', 'Ring',
          'Necklace and Chains', 'Bangle', 'Bracelet', 'Cufflinks', 'Tie']:
    ARTICLE_TO_CATEGORY[t] = 'accessories'

# Category 4: Bottomwear (lower body + full body — jeans, dresses, sarees)
for t in ['Jeans', 'Shorts', 'Trousers', 'Track Pants', 'Dresses', 'Sarees',
          'Skirts', 'Capris', 'Leggings', 'Churidar', 'Patiala', 'Jeggings',
          'Salwar', 'Stockings', 'Tights', 'Swimwear', 'Jumpsuit']:
    ARTICLE_TO_CATEGORY[t] = 'bottomwear'

# Category 5: Personal Care (beauty, fragrance, grooming products)
for t in ['Perfume and Body Mist', 'Deodorant', 'Nail Polish', 'Lipstick',
          'Lip Gloss', 'Foundation and Primer', 'Mascara', 'Lip Liner',
          'Eye Cream', 'Lip Care', 'Kajal and Eyeliner', 'Compact',
          'Sunscreen', 'Face Moisturisers', 'Body Lotion',
          'Face Wash and Cleanser', 'Hair Colour', 'Fragrance Gift Set',
          'Body Wash and Scrub']:
    ARTICLE_TO_CATEGORY[t] = 'personal_care'

CATEGORIES = ['apparel', 'footwear', 'accessories', 'bottomwear', 'personal_care']


def load_metadata(csv_path: str) -> List[Dict]:
    """Load and filter the styles.csv metadata file."""
    rows = []
    skipped = 0
    
    with open(csv_path, 'r', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            article_type = row.get('articleType', '')
            if article_type in ARTICLE_TO_CATEGORY:
                row['category'] = ARTICLE_TO_CATEGORY[article_type]
                rows.append(row)
            else:
                skipped += 1
    
    print(f"Loaded {len(rows)} images ({skipped} skipped — niche categories)")
    return rows


def split_data(
    rows: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/val/test with stratified sampling.
    
    CONCEPT: Stratified Split
    --------------------------
    Regular random split might give you 95% apparel in training and
    5% apparel in validation. Stratified split ensures EACH category
    has the same train/val/test ratio.
    
    This is critical because:
    - Validation must represent ALL categories fairly
    - Test set must be a reliable measure of real-world performance
    - Small classes (personal_care) need guaranteed representation
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    random.seed(seed)
    
    # Group by category
    by_category = defaultdict(list)
    for row in rows:
        by_category[row['category']].append(row)
    
    train, val, test = [], [], []
    
    for category, items in by_category.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    
    return train, val, test


def copy_images(
    rows: List[Dict],
    source_dir: str,
    output_dir: str,
    split_name: str
) -> Tuple[int, int]:
    """Copy images to the organized directory structure."""
    copied = 0
    missing = 0
    
    for row in rows:
        image_id = row['id']
        category = row['category']
        
        # Kaggle dataset stores images as {id}.jpg in images/ folder
        src_path = Path(source_dir) / 'images' / f"{image_id}.jpg"
        
        if not src_path.exists():
            missing += 1
            continue
        
        dst_dir = Path(output_dir) / split_name / category
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / f"{image_id}.jpg"
        
        shutil.copy2(src_path, dst_path)
        copied += 1
    
    return copied, missing


def print_summary(output_dir: str) -> None:
    """Print the final dataset summary."""
    print("\n" + "=" * 65)
    print("  DATASET PREPARATION COMPLETE")
    print("=" * 65)
    
    total = 0
    for split in ['train', 'val', 'test']:
        split_dir = Path(output_dir) / split
        if not split_dir.exists():
            continue
        
        print(f"\n  {split.upper()}/")
        split_total = 0
        for cat_dir in sorted(split_dir.iterdir()):
            if cat_dir.is_dir():
                count = len(list(cat_dir.glob('*.jpg')))
                bar = '█' * (count // 200)
                print(f"    {cat_dir.name:<18} {count:>6} images  {bar}")
                split_total += count
        print(f"    {'TOTAL':<18} {split_total:>6}")
        total += split_total
    
    print(f"\n  GRAND TOTAL: {total:>6} images")
    print(f"  Categories:  {len(CATEGORIES)}")
    print(f"  Split:       80% / 10% / 10%")
    print("\n" + "=" * 65)
    print(f"\n  Next step: python src/training/train.py \\")
    print(f"    --train-dir {output_dir}/train \\")
    print(f"    --val-dir {output_dir}/val \\")
    print(f"    --model-dir ./model \\")
    print(f"    --epochs 15 --batch-size 32")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Kaggle Fashion Product Images for training'
    )
    parser.add_argument(
        '--source', required=True,
        help='Path to extracted Kaggle dataset (contains styles.csv and images/)'
    )
    parser.add_argument(
        '--output', default='./data',
        help='Output directory for organized dataset (default: ./data)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducible splits'
    )
    
    args = parser.parse_args()
    
    # Validate source directory
    csv_path = Path(args.source) / 'styles.csv'
    images_dir = Path(args.source) / 'images'
    
    if not csv_path.exists():
        print(f"ERROR: styles.csv not found at {csv_path}")
        print(f"Download from: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small")
        return
    
    if not images_dir.exists():
        print(f"ERROR: images/ folder not found at {images_dir}")
        return
    
    image_count = len(list(images_dir.glob('*.jpg')))
    print(f"Found {image_count} images in {images_dir}")
    
    # Load metadata
    print("\n[1/3] Loading metadata...")
    rows = load_metadata(str(csv_path))
    
    # Print category distribution
    dist = Counter(row['category'] for row in rows)
    print("\nCategory distribution:")
    for cat, count in dist.most_common():
        print(f"  {cat:<18} {count:>6}")
    
    # Split data
    print("\n[2/3] Splitting into train/val/test (80/10/10)...")
    train, val, test = split_data(rows, seed=args.seed)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Copy images
    print(f"\n[3/3] Copying images to {args.output}/...")
    
    for split_name, split_data_list in [('train', train), ('val', val), ('test', test)]:
        copied, missing = copy_images(
            split_data_list, args.source, args.output, split_name
        )
        print(f"  {split_name}: {copied} copied, {missing} missing")
    
    # Print summary
    print_summary(args.output)


if __name__ == '__main__':
    main()
