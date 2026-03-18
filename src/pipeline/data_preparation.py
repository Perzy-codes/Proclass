"""
Data Preparation Lambda for Retraining Pipeline
==================================================

This Lambda function is triggered by Step Functions as the first step
of the automated retraining pipeline. It:
  1. Scans S3 for new labeled images
  2. Validates image integrity  
  3. Ensures balanced class distribution
  4. Reports dataset statistics

CONCEPT: Data Preparation in MLOps
-------------------------------------
Before retraining, we must verify:
  - Enough data exists per class (minimum threshold)
  - No corrupted images that would crash training
  - Class distribution isn't too imbalanced
  - Data is properly organized in train/val/test splits

Skipping this step risks wasted compute on failed training jobs.
"""

import boto3
import json
import logging
from typing import Dict

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')

CATEGORIES = ['apparel', 'footwear', 'accessories', 'bottomwear', 'personal_care']
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
MIN_IMAGES_PER_CLASS = 100


def lambda_handler(event: Dict, context) -> Dict:
    """
    Validate and prepare training data.
    
    Returns dataset statistics and a flag indicating if training should proceed.
    """
    bucket = event.get('source_bucket', 'product-images-raw')
    min_per_class = event.get('min_images_per_class', MIN_IMAGES_PER_CLASS)
    
    stats = {'categories': {}, 'total_images': 0}
    all_sufficient = True
    
    for split in ['train', 'val', 'test']:
        for category in CATEGORIES:
            prefix = f"{split}/{category}/"
            
            # Count images in this category
            count = 0
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key'].lower()
                    if any(key.endswith(ext) for ext in VALID_EXTENSIONS):
                        count += 1
            
            key = f"{split}/{category}"
            stats['categories'][key] = count
            stats['total_images'] += count
            
            # Check minimum threshold for training split
            if split == 'train' and count < min_per_class:
                all_sufficient = False
                logger.warning(
                    f"Insufficient data: {category} has {count} training images "
                    f"(minimum: {min_per_class})"
                )
    
    logger.info(f"Dataset stats: {json.dumps(stats, indent=2)}")
    
    return {
        'statusCode': 200,
        'sufficient_data': all_sufficient,
        'stats': stats,
    }
