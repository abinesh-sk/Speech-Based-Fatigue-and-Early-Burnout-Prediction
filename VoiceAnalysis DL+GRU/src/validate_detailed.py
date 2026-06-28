from __future__ import annotations

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score
)
from tqdm import tqdm

from .train import make_loaders
from .model import CRNN
from .data_loader import POLARITY_TO_ID


def evaluate_detailed(model, loader, device, class_names):
    """
    Evaluate model and collect all predictions and ground truth labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\n" + "="*60)
    print("Running evaluation...")
    print("="*60)
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", ncols=80):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_preds, all_labels


def print_metrics(y_true, y_pred, class_names):
    """
    Print comprehensive metrics to terminal.
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'Overall Accuracy:':<30} {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Cohen's Kappa (agreement metric)
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"{'Cohen\'s Kappa:':<30} {kappa:.4f}")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    print("\n" + "-"*60)
    print("PER-CLASS METRICS")
    print("-"*60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*60)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    print("-"*60)
    print(f"{'Macro Avg':<15} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f} {len(y_true):<10}")
    print(f"{'Weighted Avg':<15} {precision_weighted:<12.4f} {recall_weighted:<12.4f} {f1_weighted:<12.4f} {len(y_true):<10}")
    print("-"*60)
    
    # Detailed classification report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """
    Generate and save confusion matrix visualization.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with two subplots: counts and percentages
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Confusion Matrix with counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        ax=axes[0],
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', pad=20)
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Plot 2: Confusion Matrix with percentages (normalized)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'},
        ax=axes[1],
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold', pad=20)
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'Confusion matrix saved to:':<30} {output_path}")
    
    # Also print confusion matrix to terminal
    print("\n" + "="*60)
    print("CONFUSION MATRIX (Counts)")
    print("="*60)
    print(f"{'':>15}", end='')
    for name in class_names:
        print(f"{name:>15}", end='')
    print()
    print("-"*60)
    for i, name in enumerate(class_names):
        print(f"{name:>15}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>15}", end='')
        print()
    
    plt.close()


def run_detailed_eval(
    checkpoint: str,
    split: str,
    batch_size: int = 16,
    num_workers: int = 0,
    hidden_size: int = 192,
    dropout: float = 0.3,
    output_dir: str = "."
):
    """
    Run detailed evaluation with metrics and confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load data
    loaders = make_loaders(batch_size=batch_size, num_workers=num_workers)
    loader = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[split]
    
    # Class names in order of IDs
    id_to_polarity = {v: k for k, v in POLARITY_TO_ID.items()}
    class_names = [id_to_polarity[i] for i in range(len(POLARITY_TO_ID))]
    
    # Load model
    print(f"Loading model from: {checkpoint}")
    model = CRNN(
        n_mels=80,
        n_classes=len(POLARITY_TO_ID),
        hidden_size=hidden_size,
        dropout=dropout
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    
    # Evaluate
    y_pred, y_true = evaluate_detailed(model, loader, device, class_names)
    
    # Print metrics
    print_metrics(y_true, y_pred, class_names)
    
    # Generate and save confusion matrix
    output_path = Path(output_dir) / f"confusion_matrix_{split}.png"
    plot_confusion_matrix(y_true, y_pred, class_names, output_path)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Detailed model evaluation with metrics and confusion matrix")
    parser.add_argument("--checkpoint", type=str, default="crnn_best.pt", help="Path to model checkpoint")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--hidden_size", type=int, default=192, help="GRU hidden size")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save confusion matrix image")
    
    args = parser.parse_args()
    
    run_detailed_eval(
        checkpoint=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()


__all__ = ["run_detailed_eval"]
