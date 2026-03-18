"""
SageMaker Inference Handlers
==============================

These four functions define how SageMaker handles inference requests.
SageMaker calls them in order: model_fn → input_fn → predict_fn → output_fn

CONCEPT: SageMaker Inference Pipeline
---------------------------------------
When a client sends a request to the SageMaker endpoint:

  Client Request (image bytes)
       │
       ▼
  ┌──────────────────────────┐
  │ model_fn(model_dir)       │  ← Called ONCE at startup (loads model into memory)
  └──────────────────────────┘
       │
       ▼
  ┌──────────────────────────┐
  │ input_fn(request, type)   │  ← Called PER REQUEST (deserialize input)
  └──────────────────────────┘
       │
       ▼
  ┌──────────────────────────┐
  │ predict_fn(data, model)   │  ← Called PER REQUEST (run inference)
  └──────────────────────────┘
       │
       ▼
  ┌──────────────────────────┐
  │ output_fn(prediction, type)│  ← Called PER REQUEST (serialize output)
  └──────────────────────────┘
       │
       ▼
  JSON Response to Client

CONCEPT: Why Separate Functions?
---------------------------------
Separation of concerns:
  - model_fn: Heavy lifting (loading weights) — done once at startup
  - input_fn: Handles different input formats (JPEG, PNG, JSON)
  - predict_fn: Pure ML logic — model forward pass
  - output_fn: Formats response for the client

This pattern makes it easy to:
  - Support multiple input formats
  - Add preprocessing without touching model code
  - Change output format without retraining
  - Test each step independently

CONCEPT: Cold Start vs. Warm Inference
----------------------------------------
First request (cold start):
  model_fn runs → loads model → ~5-10 seconds
  
Subsequent requests (warm):
  Only input_fn → predict_fn → output_fn → ~100-200ms

SageMaker keeps the model loaded in memory between requests.
This is why we pay for the endpoint even when it's idle.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))


def model_fn(model_dir: str) -> torch.nn.Module:
    """
    Load the trained model from disk.
    
    Called ONCE when the endpoint starts up. The model stays in memory
    and is reused for all subsequent predictions.
    
    Args:
        model_dir: Directory where model artifacts are stored.
                   SageMaker downloads model.tar.gz from S3 and
                   extracts it here.
    
    Returns:
        The loaded PyTorch model in evaluation mode.
    
    CONCEPT: model.eval() is CRITICAL
    -----------------------------------
    model.eval() changes the behavior of:
    - Dropout: Disabled (all neurons active, scaled by p)
    - BatchNorm: Uses running statistics (not batch statistics)
    
    Forgetting model.eval() is a COMMON BUG that causes:
    - Different results on the same input (dropout is random)
    - Worse accuracy (dropout is hurting, not helping)
    - Inconsistent behavior between requests
    """
    from training.model import ProductClassifier
    
    logger.info(f"Loading model from {model_dir}")
    
    # Load model metadata
    info_path = os.path.join(model_dir, 'model_info.json')
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    # Create model architecture (must match training)
    model = ProductClassifier(
        num_classes=model_info['num_classes'],
        pretrained=False  # Don't download ImageNet weights — we have our own
    )
    
    # Load trained weights
    model_path = os.path.join(model_dir, 'model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    
    model.to(device)
    model.eval()  # CRITICAL: Must set to evaluation mode!
    
    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"Categories: {model_info['categories']}")
    
    return model


def input_fn(request_body: bytes, content_type: str) -> torch.Tensor:
    """
    Deserialize and preprocess the incoming request.
    
    Handles:
    - application/x-image: Raw image bytes (JPEG, PNG)
    - application/json: Base64-encoded image in JSON
    
    Args:
        request_body: Raw bytes of the request
        content_type: MIME type of the request
    
    Returns:
        Preprocessed image tensor ready for the model.
        Shape: (1, 3, 224, 224) — batch of 1 image
    
    CONCEPT: Preprocessing Must Match Training
    --------------------------------------------
    The EXACT same transformations applied during training validation
    must be applied during inference:
    
    Training val: Resize(256) → CenterCrop(224) → ToTensor() → Normalize()
    Inference:    Resize(256) → CenterCrop(224) → ToTensor() → Normalize()
    
    If you use different transforms, the model gets input in a "different format"
    than it was trained on, and accuracy drops significantly.
    
    Common mistake: Forgetting normalization during inference → model outputs garbage.
    """
    # Define inference transforms (MUST match validation transforms from training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if content_type == 'application/x-image':
        # Direct image bytes
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
    
    elif content_type == 'application/json':
        # JSON with base64-encoded image
        import base64
        data = json.loads(request_body)
        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    else:
        raise ValueError(
            f"Unsupported content type: {content_type}. "
            f"Use 'application/x-image' or 'application/json'"
        )
    
    # Apply transforms and add batch dimension
    # unsqueeze(0) adds batch dimension: (3, 224, 224) → (1, 3, 224, 224)
    # The model expects a BATCH of images, even if it's just one
    tensor = transform(image).unsqueeze(0)
    
    return tensor


def predict_fn(data: torch.Tensor, model: torch.nn.Module) -> dict:
    """
    Run inference on preprocessed data.
    
    Args:
        data: Preprocessed image tensor from input_fn
        model: Loaded model from model_fn
    
    Returns:
        Dictionary with prediction results.
    
    CONCEPT: torch.no_grad()
    -------------------------
    During inference, we don't need gradients (no training happening).
    torch.no_grad() disables gradient computation:
    - Reduces memory usage (no gradient tensors stored)
    - Speeds up forward pass (skips gradient tracking)
    """
    device = next(model.parameters()).device
    data = data.to(device)
    
    with torch.no_grad():
        output = model(data)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    probs = probabilities.cpu().numpy()[0]  # First (only) item in batch
    
    categories = ['apparel', 'footwear', 'accessories', 'bottomwear', 'personal_care']
    results = [
        {'category': cat, 'probability': float(prob)}
        for cat, prob in zip(categories, probs)
    ]
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return results


def output_fn(prediction: list, accept: str) -> str:
    """
    Serialize the prediction result.
    
    Args:
        prediction: Results from predict_fn
        accept: Desired response content type
    
    Returns:
        JSON string of predictions.
    
    CONCEPT: Content Negotiation
    -----------------------------
    The `accept` header tells us what format the client wants.
    We default to JSON because it's the standard for REST APIs.
    You could add support for CSV, Protobuf, etc. for different clients.
    """
    return json.dumps(prediction)
