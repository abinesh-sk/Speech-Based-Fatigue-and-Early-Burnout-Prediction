from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from torch.utils.data import Subset

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .class_weights import compute_class_weights
from .config import PROCESSED_DATA_DIR
from .data_loader import MelDataset, collate_pad, POLARITY_TO_ID
from .model import CRNN


def get_all_data():
    manifests = {
        split: PROCESSED_DATA_DIR / "features" / split / "manifest.json"
        for split in ("train", "val", "test")
    }
    all_items = []
    
    # Check if files exist to avoid FileNotFoundError locally vs colab 
    if manifests["train"].exists():
        for split in ("train", "val", "test"):
            # read and combine
            with manifests[split].open() as f:
                all_items.extend(json.load(f))
                
    # Create a unified dummy manifest for the dataset class to read
    # Instead of rewriting MelDataset, we'll write a temporary combined manifest
    combo_path = PROCESSED_DATA_DIR / "features" / "combined_manifest.json"
    with combo_path.open("w") as f:
        json.dump(all_items, f)
        
    full_ds = MelDataset(combo_path)
    return full_ds, all_items


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = spec_augment(x)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            if return_preds:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
    if return_preds:
        return total_loss / total, correct / total, all_labels, all_preds
    return total_loss / total, correct / total


def spec_augment(x: torch.Tensor, freq_mask_param: int = 8, time_mask_param: int = 20, noise_std: float = 0.005) -> torch.Tensor:
    """Apply simple SpecAugment-style masks and light noise during training."""
    b, n_mels, t = x.shape
    device = x.device
    # Frequency mask
    if freq_mask_param > 0:
        f_widths = torch.randint(0, freq_mask_param + 1, (b,), device=device)
        for i in range(b):
            fw = int(f_widths[i].item())
            if fw > 0 and n_mels > fw:
                f0 = int(torch.randint(0, n_mels - fw, (1,), device=device).item())
                x[i, f0 : f0 + fw, :] = 0
    # Time mask
    if time_mask_param > 0:
        t_widths = torch.randint(0, time_mask_param + 1, (b,), device=device)
        for i in range(b):
            tw = int(t_widths[i].item())
            if tw > 0 and t > tw:
                t0 = int(torch.randint(0, t - tw, (1,), device=device).item())
                x[i, :, t0 : t0 + tw] = 0
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    return x


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    full_ds, all_items = get_all_data()

    n_classes = len(POLARITY_TO_ID)

    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Use K-Fold
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = {}
    best_overall_acc = 0.0
    best_model_state = None
    
    print(f"Starting {k_folds}-Fold Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_ds)):
        print(f"\\n--- Fold {fold + 1}/{k_folds} ---")
        
        train_sub = Subset(full_ds, train_idx)
        val_sub = Subset(full_ds, val_idx)
        
        # Compute weights for the training subset specifically
        train_labels = [all_items[i]["polarity"] for i in train_idx]
        
        # Calculate class counts
        labels_count = {"negative": 0, "neutral": 0, "positive": 0}
        for lbl in train_labels:
            if lbl in labels_count:
                labels_count[lbl] += 1
                
        # Total samples in training set
        total_samples = len(train_labels)
        
        # Standard inverse frequency weights
        weights_dict = {
            c: total_samples / (len(labels_count) * max(count, 1))
            for c, count in labels_count.items()
        }
        
        sample_weights = [weights_dict[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_sub, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_pad, num_workers=args.num_workers)
        val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad, num_workers=args.num_workers)
        
        # Init model per fold
        model = CRNN(n_mels=80, n_classes=n_classes, hidden_size=args.hidden_size, dropout=args.dropout).to(device)
        
        # Loss uses the same calculated weights
        weight_tensor = torch.tensor([weights_dict[c] for c in ("negative", "neutral", "positive")], dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
        
        best_fold_acc = 0.0
        fold_best_state = None
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                fold_best_state = model.state_dict()
                
            scheduler.step()
            
        print(f"Fold {fold + 1} Best Validation Accuracy: {best_fold_acc:.4f}")
        
        fold_results[f"Fold_{fold+1}"] = {
            "best_val_acc": best_fold_acc,
            "history": history
        }
        
        if best_fold_acc > best_overall_acc:
            best_overall_acc = best_fold_acc
            best_model_state = fold_best_state
        
    print(f"\\n--- K-Fold Complete ---")
    print(f"Best overall cross-validation accuracy: {best_overall_acc:.4f}")
    
    # Save the absolute best model overall
    torch.save(best_model_state, models_dir / "crnn_best2.pt")
    
    # Optional: we can compute final test results on the whole dataset to generate overall reports
    # Load best model
    model = CRNN(n_mels=80, n_classes=n_classes, hidden_size=args.hidden_size, dropout=args.dropout).to(device)
    model.load_state_dict(best_model_state)


    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png')
    plt.close()

    # Here we show test metrics based on the full overall combination
    test_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad, num_workers=args.num_workers)
    
    test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, criterion, device, return_preds=True)
    print(f"overall_dataset_loss={test_loss:.4f} overall_acc={test_acc:.3f}")

    # Plot loss and accuracy curves (showing best fold as representative)
    best_fold_idx = 1
    best_acc = 0
    for fold_k, res in fold_results.items():
        if res["best_val_acc"] > best_acc:
            best_acc = res["best_val_acc"]
            best_fold_idx = int(fold_k.split("_")[-1])
            
    best_history = fold_results[f"Fold_{best_fold_idx}"]["history"]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(best_history['train_loss'], label='Train Loss')
    plt.plot(best_history['val_loss'], label='Val Loss')
    plt.title(f'Loss (Best Fold: {best_fold_idx})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(best_history['train_acc'], label='Train Accuracy')
    plt.plot(best_history['val_acc'], label='Val Accuracy')
    plt.title(f'Accuracy (Best Fold: {best_fold_idx})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / 'kfold_best_training_curves.png')
    plt.close()

    # Save results
    classes = ["negative", "neutral", "positive"]
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Overall Confusion Matrix (Best K-Fold Model on entire combined DS)')
    plt.tight_layout()
    plt.savefig(results_dir / 'kfold_confusion_matrix.png')
    plt.close()
    
    # Classification Report
    report = classification_report(test_labels, test_preds, target_names=classes, digits=4)
    with open(results_dir / 'metrics2.txt', 'w') as f:
        f.write(f"5-Fold CV Overall Best Validation Accuracy: {best_overall_acc:.4f}\n")
        f.write(f"Evaluating Best Model on Full Combined Dataset:\n")
        f.write(f"Overall Dataset Loss: {test_loss:.4f}\n")
        f.write(f"Overall Dataset Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    print(f"Model saved to {models_dir / 'crnn_best2.pt'}")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="crnn_best.pt")
    main(parser.parse_args())


__all__ = ["main"]
