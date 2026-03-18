"""
Test Suite for Product Classifier
===================================

CONCEPT: Testing ML Systems
------------------------------
ML systems need DIFFERENT tests than traditional software:

1. Unit Tests: Does each function work correctly?
   - Model outputs correct shapes
   - Preprocessing produces valid tensors
   - API handler returns proper responses

2. Integration Tests: Do components work together?
   - Can we load a saved model and make predictions?
   - Does the Lambda handler → SageMaker pipeline work?

3. ML-Specific Tests:
   - Does the model output valid probabilities (sum to 1)?
   - Is the model deterministic in eval mode?
   - Does training actually reduce the loss?
   - Are predictions reasonable (not always the same class)?

CONCEPT: pytest
----------------
pytest is the standard Python testing framework.
Key features:
  - @pytest.fixture: Setup reusable test data
  - @pytest.mark: Tag tests (e.g., mark.slow for long-running tests)
  - assert: Simple assertion syntax
  - conftest.py: Shared fixtures across test files
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
from unittest.mock import patch, MagicMock
from PIL import Image
import io
import base64

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'training'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'inference'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))


# ============================================================================
# FIXTURES — Reusable test components
# ============================================================================
# CONCEPT: Fixtures
# Fixtures create shared test data/objects. Each test function that needs
# a model gets a fresh instance without duplicating creation code.

@pytest.fixture
def model():
    """Create a fresh model instance for testing."""
    from model import ProductClassifier
    return ProductClassifier(num_classes=5, pretrained=False)


@pytest.fixture
def sample_batch():
    """Create a batch of random images for testing."""
    # Shape: (batch_size=4, channels=3, height=224, width=224)
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_image_bytes():
    """Create a sample JPEG image as bytes."""
    img = Image.new('RGB', (300, 300), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def sample_api_event(sample_image_bytes):
    """Create a mock API Gateway event."""
    return {
        'body': json.dumps({
            'image': base64.b64encode(sample_image_bytes).decode('utf-8'),
            'product_id': 'TEST-001'
        }),
        'headers': {'Content-Type': 'application/json'},
        'httpMethod': 'POST',
    }


# ============================================================================
# MODEL TESTS
# ============================================================================

class TestProductClassifier:
    """Tests for the ProductClassifier model."""
    
    def test_model_creation(self, model):
        """Model should initialize without errors."""
        assert model is not None
        assert model.num_classes == 5
    
    def test_output_shape(self, model, sample_batch):
        """
        Model should output (batch_size, num_classes) tensor.
        
        This is the MOST IMPORTANT test. If the output shape is wrong,
        nothing downstream will work (loss computation, predictions, etc.)
        """
        output = model(sample_batch)
        assert output.shape == (4, 5), f"Expected (4, 5), got {output.shape}"
    
    def test_output_is_logits(self, model, sample_batch):
        """
        Output should be raw logits (not probabilities).
        
        Logits can be any real number (positive or negative).
        Probabilities are between 0 and 1 and sum to 1.
        We want logits because CrossEntropyLoss applies softmax internally.
        """
        output = model(sample_batch)
        # Logits can be negative and don't sum to 1
        assert output.min() < 0 or output.max() > 1 or abs(output.sum(dim=1).mean() - 1.0) > 0.01
    
    def test_softmax_produces_valid_probabilities(self, model, sample_batch):
        """After softmax, output should be valid probabilities."""
        output = model(sample_batch)
        probs = torch.nn.functional.softmax(output, dim=1)
        
        # All probabilities should be between 0 and 1
        assert (probs >= 0).all(), "Probabilities must be non-negative"
        assert (probs <= 1).all(), "Probabilities must be <= 1"
        
        # Probabilities should sum to ~1 for each sample
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5), \
            f"Probabilities should sum to 1, got {sums}"
    
    def test_predict_method(self, model, sample_batch):
        """predict() should return class indices and confidence scores."""
        predicted_classes, confidence = model.predict(sample_batch)
        
        assert predicted_classes.shape == (4,)
        assert confidence.shape == (4,)
        
        # Classes should be valid indices (0-4)
        assert (predicted_classes >= 0).all()
        assert (predicted_classes < 5).all()
        
        # Confidence should be between 0 and 1
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()
    
    def test_eval_mode_deterministic(self, model, sample_batch):
        """
        In eval mode, same input should produce same output.
        
        This tests that dropout is properly disabled.
        If dropout were active, outputs would differ between calls.
        """
        model.eval()
        with torch.no_grad():
            output1 = model(sample_batch)
            output2 = model(sample_batch)
        
        assert torch.equal(output1, output2), \
            "Model should be deterministic in eval mode"
    
    def test_train_mode_has_gradients(self, model, sample_batch):
        """
        In train mode, forward pass should create gradient-ready tensors.
        This ensures backpropagation will work.
        """
        model.train()
        output = model(sample_batch)
        assert output.requires_grad, "Output should have gradients in train mode"
    
    def test_frozen_layers_dont_require_grad(self, model):
        """Early layers should be frozen (requires_grad=False)."""
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        
        assert frozen_count > 0, "Some layers should be frozen"
        assert trainable_count > 0, "Some layers should be trainable"
        assert frozen_count < sum(1 for _ in model.parameters()), \
            "Not ALL layers should be frozen"
    
    def test_parameter_count(self, model):
        """Model should have expected parameter counts."""
        total = model.get_total_params()
        trainable = model.get_trainable_params()
        
        assert total > 10_000_000, "ResNet18 should have ~11.7M params"
        assert total < 15_000_000, "Should be ResNet18, not ResNet50"
        assert trainable < total, "Some params should be frozen"
        assert trainable > 100_000, "Need enough trainable params"
    
    def test_categories_constant(self, model):
        """Categories should be the expected product types."""
        expected = ['apparel', 'footwear', 'accessories', 'bottomwear', 'personal_care']
        assert model.CATEGORIES == expected
    
    def test_single_image_inference(self, model):
        """Model should handle a single image (batch_size=1)."""
        single_image = torch.randn(1, 3, 224, 224)
        output = model(single_image)
        assert output.shape == (1, 5)


# ============================================================================
# PREPROCESSING TESTS
# ============================================================================

class TestPreprocessing:
    """Tests for image preprocessing."""
    
    def test_train_transforms_output_shape(self):
        """Training transforms should produce (3, 224, 224) tensor."""
        from dataset import get_train_transforms
        transform = get_train_transforms()
        
        img = Image.new('RGB', (500, 500), color='blue')
        tensor = transform(img)
        
        assert tensor.shape == (3, 224, 224)
    
    def test_val_transforms_output_shape(self):
        """Validation transforms should produce (3, 224, 224) tensor."""
        from dataset import get_val_transforms
        transform = get_val_transforms()
        
        img = Image.new('RGB', (500, 500), color='blue')
        tensor = transform(img)
        
        assert tensor.shape == (3, 224, 224)
    
    def test_val_transforms_deterministic(self):
        """Validation transforms should give same result every time."""
        from dataset import get_val_transforms
        transform = get_val_transforms()
        
        img = Image.new('RGB', (500, 500), color='green')
        tensor1 = transform(img)
        tensor2 = transform(img)
        
        assert torch.equal(tensor1, tensor2), \
            "Validation transforms must be deterministic"
    
    def test_normalization_applied(self):
        """After normalization, values should NOT be in [0, 1] range."""
        from dataset import get_val_transforms
        transform = get_val_transforms()
        
        img = Image.new('RGB', (300, 300), color='white')
        tensor = transform(img)
        
        # After ImageNet normalization, white (1.0) becomes:
        # (1.0 - 0.485) / 0.229 ≈ 2.25 for red channel
        assert tensor.max() > 1.0, "Normalization should produce values > 1"
    
    def test_handles_grayscale_image(self):
        """Should handle grayscale images by converting to RGB."""
        from dataset import get_val_transforms
        transform = get_val_transforms()
        
        # Grayscale image (mode 'L')
        img = Image.new('L', (300, 300), color=128)
        img = img.convert('RGB')  # Dataset does this conversion
        tensor = transform(img)
        
        assert tensor.shape == (3, 224, 224)
    
    def test_handles_rgba_image(self):
        """Should handle RGBA images by converting to RGB."""
        from dataset import get_val_transforms
        transform = get_val_transforms()
        
        img = Image.new('RGBA', (300, 300), color=(255, 0, 0, 128))
        img = img.convert('RGB')
        tensor = transform(img)
        
        assert tensor.shape == (3, 224, 224)


# ============================================================================
# TRAINING TESTS
# ============================================================================

class TestTraining:
    """Tests for the training pipeline."""
    
    def test_loss_decreases(self, model):
        """
        One step of training should reduce the loss.
        
        This is a basic sanity check that backpropagation works.
        If loss doesn't decrease, something is fundamentally wrong
        (wrong loss function, frozen layers, etc.)
        """
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.01  # High LR for quick test
        )
        
        # Create dummy data
        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 5, (8,))
        
        # Measure initial loss
        output1 = model(images)
        loss1 = criterion(output1, labels)
        
        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
        # Loss should be lower
        output2 = model(images)
        loss2 = criterion(output2, labels)
        
        assert loss2.item() < loss1.item(), \
            f"Loss should decrease: {loss1.item():.4f} → {loss2.item():.4f}"
    
    def test_class_weights_computation(self, tmp_path):
        """Class weights should be inversely proportional to frequency."""
        from dataset import ProductImageDataset
        
        # Create dummy dataset structure
        for cat in ['electronics', 'clothing', 'furniture', 'books', 'toys']:
            cat_dir = tmp_path / cat
            cat_dir.mkdir()
            # Different number of images per class
            num_images = {'electronics': 100, 'clothing': 50, 'furniture': 30, 
                         'books': 10, 'toys': 10}
            for i in range(num_images[cat]):
                img = Image.new('RGB', (10, 10))
                img.save(cat_dir / f'{i}.jpg')
        
        dataset = ProductImageDataset(str(tmp_path))
        weights = dataset.get_class_weights()
        
        # Minority classes (books, toys) should have HIGHER weight
        assert weights[3] > weights[0], "Books (minority) should have higher weight than electronics (majority)"
        assert weights[4] > weights[1], "Toys (minority) should have higher weight than clothing"


# ============================================================================
# LAMBDA HANDLER TESTS
# ============================================================================

class TestLambdaHandler:
    """Tests for the Lambda function."""
    
    def test_validate_request_missing_image(self):
        """Should reject requests without image field."""
        from lambda_handler import validate_request
        errors, _ = validate_request({})
        assert len(errors) > 0
        assert 'image' in errors[0].lower()
    
    def test_validate_request_valid(self, sample_image_bytes):
        """Should accept valid requests."""
        from lambda_handler import validate_request
        body = {
            'image': base64.b64encode(sample_image_bytes).decode('utf-8'),
            'product_id': 'TEST-001'
        }
        errors, product_id = validate_request(body)
        assert len(errors) == 0
        assert product_id == 'TEST-001'
    
    def test_preprocess_image(self, sample_image_bytes):
        """Preprocessed image should be valid JPEG at 224x224."""
        from lambda_handler import preprocess_image
        processed = preprocess_image(sample_image_bytes)
        
        # Should be valid image
        img = Image.open(io.BytesIO(processed))
        assert img.size == (224, 224)
        assert img.mode == 'RGB'
    
    @patch('lambda_handler.sagemaker_runtime')
    @patch('lambda_handler.s3')
    @patch('lambda_handler.dynamodb')
    def test_handler_success(self, mock_dynamo, mock_s3, mock_sagemaker, sample_api_event):
        """Full handler should return 200 on valid request."""
        from lambda_handler import lambda_handler
        
        # Mock SageMaker response
        mock_response = {
            'Body': MagicMock(
                read=MagicMock(return_value=json.dumps([
                    {'category': 'electronics', 'probability': 0.95},
                    {'category': 'clothing', 'probability': 0.02},
                    {'category': 'furniture', 'probability': 0.01},
                    {'category': 'books', 'probability': 0.01},
                    {'category': 'toys', 'probability': 0.01},
                ]).encode())
            )
        }
        mock_sagemaker.invoke_endpoint.return_value = mock_response
        
        # Mock DynamoDB
        mock_table = MagicMock()
        mock_dynamo.Table.return_value = mock_table
        
        # Mock context
        mock_context = MagicMock()
        mock_context.aws_request_id = 'test-request-123'
        
        result = lambda_handler(sample_api_event, mock_context)
        
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert body['prediction'] == 'electronics'
        assert body['confidence'] == 0.95
    
    def test_handler_invalid_request(self):
        """Handler should return 400 for invalid requests."""
        from lambda_handler import lambda_handler
        
        event = {'body': json.dumps({})}
        mock_context = MagicMock()
        mock_context.aws_request_id = 'test-123'
        
        result = lambda_handler(event, mock_context)
        assert result['statusCode'] == 400


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests (run with: pytest -m integration)."""
    
    def test_model_save_and_load(self, model, tmp_path):
        """Model should produce same results after save/load cycle."""
        # Save
        model_path = tmp_path / 'model.pth'
        torch.save(model.state_dict(), model_path)
        
        # Load
        from model import ProductClassifier
        loaded_model = ProductClassifier(num_classes=5, pretrained=False)
        loaded_model.load_state_dict(torch.load(model_path))
        loaded_model.eval()
        model.eval()
        
        # Compare outputs
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = loaded_model(test_input)
        
        assert torch.allclose(original_output, loaded_output, atol=1e-6), \
            "Loaded model should produce identical results"
    
    def test_end_to_end_prediction(self, model):
        """Full pipeline: raw image → preprocess → predict → result."""
        from dataset import get_val_transforms
        
        model.eval()
        transform = get_val_transforms()
        
        # Simulate a product image
        img = Image.new('RGB', (640, 480), color='blue')
        tensor = transform(img).unsqueeze(0)
        
        # Get prediction
        predicted_class, confidence = model.predict(tensor)
        
        # Verify results
        categories = ProductClassifier.CATEGORIES
        pred_category = categories[predicted_class.item()]
        
        assert pred_category in categories
        assert 0.0 <= confidence.item() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
