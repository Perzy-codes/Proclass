"""
Model Evaluation & Metrics
===========================

Comprehensive evaluation of the trained model with metrics that matter
for production ML systems.

CONCEPT: Why Multiple Metrics?
-------------------------------
Accuracy alone is NOT enough. Consider:
  - Dataset: 90% electronics, 10% books
  - Model predicts "electronics" for EVERYTHING
  - Accuracy = 90%! But the model is useless for books.

That's why we use:
  - Precision: Of all predictions for class X, what % were correct?
  - Recall: Of all actual class X items, what % did we find?
  - F1 Score: Harmonic mean of precision and recall (balanced measure)
  - Confusion Matrix: Shows exactly where the model confuses categories

CONCEPT: Precision vs Recall
------------------------------
Precision = TP / (TP + FP) — "When I predict X, am I right?"
Recall    = TP / (TP + FN) — "Did I find all the X items?"

For e-commerce:
  - High precision matters: Don't put clothing in the electronics section
  - High recall matters: Don't miss any electronics listings
  - F1 balances both: Good for overall evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import json
import logging
from collections import defaultdict
from pathlib import Path

from model import ProductClassifier
from dataset import ProductImageDataset, get_val_transforms

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    categories: List[str]
) -> Dict:
    """
    Full evaluation with per-class metrics.
    
    CONCEPT: Confusion Matrix
    --------------------------
    A confusion matrix shows predictions vs. actual labels:
    
                        Predicted
                    Elec  Cloth  Furn  Book  Toys
    Actual  Elec  [ 95     2      1     1     1  ]  ← 95% recall for electronics
            Cloth [  1    88      3     5     3  ]
            Furn  [  2     4     90     2     2  ]
            Book  [  3     5      2    87     3  ]
            Toys  [  1     3      1     2    93  ]
    
    Diagonal = correct predictions
    Off-diagonal = errors (which classes get confused)
    
    This tells you:
    - Electronics and Toys are easy to classify
    - Clothing and Books get confused sometimes (both rectangular?)
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Overall accuracy
    accuracy = (all_predictions == all_labels).mean() * 100
    
    # Per-class metrics
    num_classes = len(categories)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true, pred in zip(all_labels, all_predictions):
        confusion_matrix[true][pred] += 1
    
    per_class_metrics = {}
    for i, category in enumerate(categories):
        tp = confusion_matrix[i][i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[category] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'support': int(confusion_matrix[i].sum()),  # Total actual samples
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }
    
    # Macro averages (treat all classes equally)
    macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
    macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
    macro_f1 = np.mean([m['f1_score'] for m in per_class_metrics.values()])
    
    # Confidence statistics
    max_probs = all_probabilities.max(axis=1)
    
    results = {
        'overall_accuracy': round(accuracy, 2),
        'macro_precision': round(macro_precision, 4),
        'macro_recall': round(macro_recall, 4),
        'macro_f1': round(macro_f1, 4),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': confusion_matrix.tolist(),
        'categories': categories,
        'total_samples': len(all_labels),
        'confidence_stats': {
            'mean': round(float(max_probs.mean()), 4),
            'median': round(float(np.median(max_probs)), 4),
            'std': round(float(max_probs.std()), 4),
            'min': round(float(max_probs.min()), 4),
            'below_80_pct': round(float((max_probs < 0.8).mean()) * 100, 2),
        }
    }
    
    return results


def print_evaluation_report(results: Dict) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Macro Precision:  {results['macro_precision']:.4f}")
    print(f"Macro Recall:     {results['macro_recall']:.4f}")
    print(f"Macro F1 Score:   {results['macro_f1']:.4f}")
    print(f"Total Samples:    {results['total_samples']}")
    
    print("\n" + "-" * 70)
    print("Per-Class Metrics:")
    print("-" * 70)
    print(f"{'Category':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    
    for cat, metrics in results['per_class_metrics'].items():
        print(
            f"{cat:<15} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1_score']:>10.4f} {metrics['support']:>10d}"
        )
    
    print("\n" + "-" * 70)
    print("Confusion Matrix:")
    print("-" * 70)
    categories = results['categories']
    cm = results['confusion_matrix']
    
    # Header
    print(f"{'':>12}", end="")
    for cat in categories:
        print(f"{cat[:6]:>8}", end="")
    print()
    
    # Rows
    for i, cat in enumerate(categories):
        print(f"{cat:<12}", end="")
        for j in range(len(categories)):
            print(f"{cm[i][j]:>8d}", end="")
        print()
    
    print("\n" + "-" * 70)
    print("Confidence Statistics:")
    print("-" * 70)
    cs = results['confidence_stats']
    print(f"  Mean confidence:    {cs['mean']:.4f}")
    print(f"  Median confidence:  {cs['median']:.4f}")
    print(f"  Std deviation:      {cs['std']:.4f}")
    print(f"  Below 80% conf:    {cs['below_80_pct']:.2f}% of predictions")
    
    print("\n" + "=" * 70)


def evaluate_from_checkpoint(
    model_dir: str,
    test_dir: str,
    device: torch.device = None
) -> Dict:
    """
    Load a saved model and evaluate on test data.
    
    This is used by the retraining pipeline to evaluate new models
    before deploying them to production.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model info
    with open(Path(model_dir) / 'model_info.json') as f:
        model_info = json.load(f)
    
    # Create model and load weights
    model = ProductClassifier(num_classes=model_info['num_classes'])
    model.load_state_dict(
        torch.load(Path(model_dir) / 'model.pth', map_location=device)
    )
    model.to(device)
    
    # Create test dataset
    test_dataset = ProductImageDataset(
        data_dir=test_dir,
        transform=get_val_transforms()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, model_info['categories'])
    print_evaluation_report(results)
    
    return results
