"""
Client-Side Prediction Helper
================================

Convenience wrapper for calling the deployed API endpoint.
Use this for testing, integration, and batch predictions.

CONCEPT: Client Libraries
---------------------------
A good project provides a client library so users don't need to 
craft raw HTTP requests. This is what AWS does with boto3 — it wraps
the raw REST API into a clean Python interface.

Usage:
    from predictor import ProductPredictor
    
    predictor = ProductPredictor("https://abc123.execute-api.us-east-1.amazonaws.com/prod")
    result = predictor.classify("path/to/product_image.jpg")
    print(result)
    # {'prediction': 'electronics', 'confidence': 0.95, ...}
"""

import base64
import json
import requests
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class ProductPredictor:
    """
    Client for the Product Classifier API.
    
    Handles image encoding, API communication, and response parsing.
    Supports both single predictions and batch processing.
    """
    
    def __init__(self, api_url: str, api_key: Optional[str] = None, timeout: int = 30):
        """
        Args:
            api_url: Full URL to the /classify endpoint
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['x-api-key'] = api_key
    
    def classify(
        self, 
        image_path: str, 
        product_id: Optional[str] = None
    ) -> Dict:
        """
        Classify a single product image.
        
        Args:
            image_path: Path to the image file
            product_id: Optional product identifier
        
        Returns:
            Dictionary with prediction, confidence, and all scores
            
        Raises:
            FileNotFoundError: If image_path doesn't exist
            requests.RequestException: If API call fails
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Build request payload
        payload = {'image': image_b64}
        if product_id:
            payload['product_id'] = product_id
        
        # Make API call
        response = requests.post(
            f"{self.api_url}/classify",
            json=payload,
            headers=self.headers,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def classify_batch(
        self, 
        image_paths: List[str],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Classify multiple images sequentially.
        
        For high-throughput batch processing, consider using
        SageMaker Batch Transform instead of the real-time API.
        
        Args:
            image_paths: List of image file paths
            verbose: Print progress
        
        Returns:
            List of prediction results
        """
        results = []
        
        for i, path in enumerate(image_paths):
            try:
                result = self.classify(path)
                results.append({
                    'image': path,
                    'status': 'success',
                    **result
                })
                if verbose:
                    print(
                        f"[{i+1}/{len(image_paths)}] {Path(path).name}: "
                        f"{result['prediction']} ({result['confidence']:.2%})"
                    )
            except Exception as e:
                results.append({
                    'image': path,
                    'status': 'error',
                    'error': str(e)
                })
                if verbose:
                    print(f"[{i+1}/{len(image_paths)}] {Path(path).name}: ERROR - {e}")
        
        return results
    
    def health_check(self) -> bool:
        """Check if the API is responding."""
        try:
            response = requests.get(
                f"{self.api_url}/health",
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False


def classify_local(image_path: str, model_dir: str = './model') -> Dict:
    """
    Classify an image using a locally saved model (no API needed).
    
    Useful for:
    - Development and debugging
    - Offline testing
    - Environments without API access
    """
    import torch
    from PIL import Image
    
    # Add training module to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
    from model import ProductClassifier
    from dataset import get_val_transforms
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProductClassifier(num_classes=5, pretrained=False)
    
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Preprocess image
    transform = get_val_transforms()
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    results = model.predict_with_all_scores(tensor)
    top_result = results[0]
    top_category = max(top_result, key=top_result.get)
    
    return {
        'prediction': top_category,
        'confidence': top_result[top_category],
        'all_predictions': [
            {'category': cat, 'probability': prob}
            for cat, prob in top_result.items()
        ]
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify a product image')
    parser.add_argument('--image', required=True, help='Path to product image')
    parser.add_argument('--api-url', default=None, help='API endpoint URL')
    parser.add_argument('--model-dir', default='./model', help='Local model directory')
    parser.add_argument('--local', action='store_true', help='Use local model instead of API')
    
    args = parser.parse_args()
    
    if args.local or not args.api_url:
        print("Using local model...")
        result = classify_local(args.image, args.model_dir)
    else:
        predictor = ProductPredictor(args.api_url)
        result = predictor.classify(args.image)
    
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nAll predictions:")
    for pred in result['all_predictions']:
        bar = '█' * int(pred['probability'] * 40)
        print(f"  {pred['category']:<15} {pred['probability']:.2%} {bar}")
