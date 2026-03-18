"""
AWS Lambda Function — Image Preprocessing & Classification
============================================================

This Lambda function is the core orchestrator. It receives image upload requests
from API Gateway, preprocesses images, calls SageMaker for prediction, and
stores results in DynamoDB.

CONCEPT: AWS Lambda
--------------------
Lambda is "serverless compute" — you write a function, AWS runs it.

How it works:
  1. A trigger (API Gateway, S3 event, etc.) invokes your function
  2. AWS allocates a container with your code and dependencies
  3. Your function runs (up to 15 min timeout)
  4. AWS bills you ONLY for execution time (per 1ms)
  5. If no requests come, you pay $0 (unlike EC2 which runs 24/7)

Key concepts:
  - Cold start: First invocation takes 1-3 seconds (container initialization)
  - Warm start: Subsequent invocations reuse the container (~100ms)
  - Concurrency: Lambda auto-scales (up to 1000 concurrent by default)
  - Memory: More memory = more CPU (1024 MB is good for image processing)

CONCEPT: Lambda Handler Pattern
---------------------------------
Every Lambda function has a handler:
  def lambda_handler(event, context):
    - event: Contains the request data (API Gateway passes the HTTP request here)
    - context: Metadata about the invocation (request ID, time remaining, etc.)
    - return: A response dict that API Gateway converts to an HTTP response

CONCEPT: Why Lambda + SageMaker (not just Lambda)?
----------------------------------------------------
Why not run the ML model directly inside Lambda?
  1. Lambda has a 10 GB deployment size limit (model + PyTorch > 2 GB)
  2. Lambda has limited memory (10 GB max) — loading a model on every cold start is slow
  3. SageMaker keeps the model in memory permanently (no cold start overhead)
  4. SageMaker supports GPU inference and auto-scaling

Lambda handles the "lightweight" work (validation, resizing, S3 storage).
SageMaker handles the "heavy" work (ML inference).
"""

import json
import boto3
import base64
from PIL import Image
import io
import uuid
import time
import logging
from typing import Dict, Any, Optional
from decimal import Decimal

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ============================================================================
# CONCEPT: Initializing AWS Clients OUTSIDE the Handler
# ============================================================================
# These clients are initialized when the Lambda container starts (cold start).
# They persist across invocations (warm starts) — we don't re-create them.
# This is called "connection reuse" and significantly speeds up warm invocations.
#
# Think of it like this:
#   Cold start: Open a phone line to S3, SageMaker, DynamoDB (expensive)
#   Warm start: Reuse the open phone lines (cheap)
# ============================================================================

s3 = boto3.client('s3')
sagemaker_runtime = boto3.client('sagemaker-runtime')
dynamodb = boto3.resource('dynamodb')

# Configuration — in production, use environment variables (set in SAM template)
PROCESSED_BUCKET = 'product-images-processed'
ENDPOINT_NAME = 'product-classifier'
TABLE_NAME = 'product-predictions'
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to accept prediction


def validate_request(body: Dict) -> tuple:
    """
    Validate the incoming request.
    
    CONCEPT: Input Validation
    --------------------------
    NEVER trust client input. Always validate:
    - Required fields exist
    - Data types are correct
    - Values are within expected ranges
    - Image is a valid format and reasonable size
    
    For ML systems, bad input leads to bad predictions (garbage in, garbage out).
    Validation catches issues early and returns clear error messages.
    """
    errors = []
    
    if 'image' not in body:
        errors.append("Missing required field: 'image' (base64-encoded)")
    
    if 'image' in body:
        try:
            image_bytes = base64.b64decode(body['image'])
            if len(image_bytes) > MAX_IMAGE_SIZE:
                errors.append(f"Image too large: {len(image_bytes)} bytes (max: {MAX_IMAGE_SIZE})")
            if len(image_bytes) < 100:
                errors.append("Image too small — likely corrupted")
        except Exception:
            errors.append("Invalid base64 encoding for 'image' field")
    
    product_id = body.get('product_id', str(uuid.uuid4()))
    
    return errors, product_id


def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Resize and optimize image for inference.
    
    CONCEPT: Why Preprocess in Lambda?
    ------------------------------------
    1. Reduce data transfer to SageMaker (smaller image = faster transfer)
    2. Standardize input size (SageMaker expects 224x224)
    3. Convert to consistent format (JPEG, RGB)
    4. Validate image is not corrupted
    
    This saves ~50ms per request compared to letting SageMaker handle it.
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (handles RGBA, grayscale, CMYK, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224 (what our model expects)
    # CONCEPT: LANCZOS resampling
    # The best quality downsampling algorithm. It uses a sinc-based filter
    # that preserves edges and details better than BILINEAR or NEAREST.
    image = image.resize((224, 224), Image.LANCZOS)
    
    # Convert back to JPEG bytes
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    
    return buffer.getvalue()


def invoke_sagemaker(image_bytes: bytes) -> list:
    """
    Call SageMaker endpoint for prediction.
    
    CONCEPT: SageMaker Runtime
    ---------------------------
    sagemaker-runtime is the service for invoking endpoints (inference).
    sagemaker is the service for managing endpoints (create/delete/update).
    
    The invoke_endpoint call:
    1. Sends image bytes to the SageMaker endpoint
    2. SageMaker routes to our inference.py handlers
    3. Returns prediction results as JSON
    
    Latency breakdown (typical):
    - Network to SageMaker: ~20ms (same region)
    - input_fn preprocessing: ~30ms
    - Model forward pass: ~50ms (CPU) or ~10ms (GPU)
    - output_fn serialization: ~1ms
    - Network return: ~20ms
    - Total: ~120ms
    """
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/x-image',
        Body=image_bytes
    )
    
    result = json.loads(response['Body'].read().decode())
    return result


def store_result(product_id: str, prediction: Dict, request_id: str) -> None:
    """
    Store prediction in DynamoDB.
    
    CONCEPT: DynamoDB
    ------------------
    DynamoDB is a fully managed NoSQL (key-value + document) database.
    
    Why DynamoDB for predictions?
    - Single-digit millisecond latency (fast reads for the UI)
    - Auto-scaling (handles traffic spikes automatically)
    - Pay-per-request (no cost when idle)
    - TTL (automatically delete old predictions)
    - Streams (trigger events when new predictions arrive)
    
    CONCEPT: Partition Key + Sort Key
    -----------------------------------
    Our table uses:
    - Partition key: product_id (groups all predictions for one product)
    - Sort key: timestamp (orders predictions by time)
    
    This allows efficient queries like:
    - "Get latest prediction for product X" (query by partition key, sort desc)
    - "Get all predictions for product X" (query by partition key)
    
    CONCEPT: DynamoDB and Decimal
    ------------------------------
    DynamoDB doesn't support Python float. We must use Decimal.
    This is a common gotcha that causes "TypeError: Float types are not supported"
    """
    table = dynamodb.Table(TABLE_NAME)
    
    # DynamoDB requires Decimal instead of float
    item = {
        'product_id': product_id,
        'timestamp': str(int(time.time() * 1000)),  # Millisecond timestamp
        'category': prediction['category'],
        'confidence': Decimal(str(round(prediction['probability'], 6))),
        'request_id': request_id,
        'ttl': int(time.time()) + (90 * 24 * 60 * 60),  # Auto-delete after 90 days
    }
    
    table.put_item(Item=item)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict:
    """
    Main Lambda handler — orchestrates the classification pipeline.
    
    CONCEPT: API Gateway Integration
    ----------------------------------
    API Gateway sends the HTTP request as the `event` dict:
    {
        "body": "{\"image\": \"base64...\", \"product_id\": \"PROD-001\"}",
        "headers": {"Content-Type": "application/json"},
        "httpMethod": "POST",
        "path": "/classify",
        ...
    }
    
    We return a dict that API Gateway converts to an HTTP response:
    {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": "{\"prediction\": \"electronics\", \"confidence\": 0.95}"
    }
    
    CONCEPT: CORS Headers
    ----------------------
    Cross-Origin Resource Sharing headers allow web browsers to call our API
    from different domains. Without these, JavaScript on yourwebsite.com
    can't call our API at aws-api.execute-api.amazonaws.com.
    """
    request_start = time.time()
    request_id = context.aws_request_id if context else str(uuid.uuid4())
    
    # Standard CORS headers
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',          # Allow any domain
        'Access-Control-Allow-Methods': 'POST, GET', # Allowed HTTP methods
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    }
    
    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        # Validate
        errors, product_id = validate_request(body)
        if errors:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': 'Validation failed',
                    'details': errors
                })
            }
        
        # Decode and preprocess image
        image_bytes = base64.b64decode(body['image'])
        processed_bytes = preprocess_image(image_bytes)
        
        # Store processed image in S3 (for audit trail and retraining)
        s3_key = f"predictions/{product_id}/{request_id}.jpg"
        s3.put_object(
            Bucket=PROCESSED_BUCKET,
            Key=s3_key,
            Body=processed_bytes,
            ContentType='image/jpeg',
            Metadata={
                'product_id': product_id,
                'request_id': request_id,
            }
        )
        
        # Call SageMaker for prediction
        predictions = invoke_sagemaker(processed_bytes)
        top_prediction = predictions[0]
        
        # Store result in DynamoDB
        store_result(product_id, top_prediction, request_id)
        
        # Calculate latency
        latency_ms = (time.time() - request_start) * 1000
        
        # Check confidence
        low_confidence = top_prediction['probability'] < CONFIDENCE_THRESHOLD
        
        logger.info(
            f"Prediction: {top_prediction['category']} "
            f"({top_prediction['probability']:.2%}) "
            f"for product {product_id} "
            f"in {latency_ms:.0f}ms"
        )
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'product_id': product_id,
                'prediction': top_prediction['category'],
                'confidence': top_prediction['probability'],
                'low_confidence_flag': low_confidence,
                'all_predictions': predictions,
                'latency_ms': round(latency_ms, 1),
                'request_id': request_id,
            })
        }
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e),
                'request_id': request_id,
            })
        }
