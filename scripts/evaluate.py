"""
Evaluation script for the Car Type Classifier.

Usage:
    python scripts/evaluate.py --model models/classifier/best_model.pth
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from classifier import CarTypeClassifier, CAR_TYPES, NUM_CLASSES
from utils import ensure_dir


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate Car Type Classifier')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='dataset/dataset/test',
                        help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                        help='Output directory for results')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_per_class_metrics(y_true, y_pred, classes, output_path):
    """Plot per-class precision, recall, F1."""
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Car Type')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Classification Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to: {output_path}")


def main():
    args = get_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    model = CarTypeClassifier(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if 'val_acc' in checkpoint:
        print(f"Model validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Data
    print(f"\nLoading test data from: {args.data_dir}")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluate
    print("\nEvaluating...")
    predictions, labels, probabilities = evaluate(model, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(labels, predictions, target_names=CAR_TYPES))
    
    # Save results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': classification_report(labels, predictions, target_names=CAR_TYPES, output_dict=True)
    }
    
    # Save as text
    with open(output_dir / 'results.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Data: {args.data_dir}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1-Score:  {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(labels, predictions, target_names=CAR_TYPES))
    
    print(f"\nResults saved to: {output_dir / 'results.txt'}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        labels, predictions, CAR_TYPES,
        output_dir / 'confusion_matrix.png'
    )
    
    # Plot per-class metrics
    plot_per_class_metrics(
        labels, predictions, CAR_TYPES,
        output_dir / 'per_class_metrics.png'
    )
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
