"""K-Fold Cross-Validation Training for the CRNN emotion classifier.

Usage
-----
    python -m src.train_kfold [--epochs N] [--batch_size B] [--hidden_size H] [--dropout D]

Output
------
    models/fold1_model.pt  … models/fold5_model.pt
    models/kfold_results.txt

The existing crnn_best.pt in the project root is NOT touched.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from .config import MODELS_DIR, PROCESSED_DATA_DIR
from .data_loader import POLARITY_TO_ID, collate_pad
from .model import CRNN
from .train import evaluate, train_one_epoch

K_FOLDS = 5


# ---------------------------------------------------------------------------
# Dataset backed by an explicit list of manifest items
# ---------------------------------------------------------------------------

class ManifestItemDataset(Dataset):
    """Thin Dataset that wraps a list of manifest dicts (already merged)."""

    def __init__(self, items: List[Dict[str, str]], max_frames: int | None = None):
        self.items = items
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.items[idx]
        feat_path = PROCESSED_DATA_DIR / Path(item["feature_path"])
        mel = np.load(feat_path)
        if self.max_frames is not None:
            mel = mel[:, : self.max_frames]
        mel_tensor = torch.from_numpy(mel.astype(np.float32))
        label = POLARITY_TO_ID[item["polarity"]]
        return mel_tensor, label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest(split: str) -> List[Dict[str, str]]:
    path = PROCESSED_DATA_DIR / "features" / split / "manifest.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_class_weights_from_items(items: List[Dict[str, str]]) -> Dict[str, float]:
    counts: Counter = Counter(item["polarity"] for item in items)
    total = sum(counts.values())
    n_classes = len(counts)
    return {lbl: total / (n_classes * cnt) for lbl, cnt in counts.items()}


def make_weighted_sampler(items: List[Dict[str, str]], weights_dict: Dict[str, float]) -> WeightedRandomSampler:
    sample_weights = [weights_dict[it["polarity"]] for it in items]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def kfold_split(n: int, k: int, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_indices, val_indices) tuples, stratified by equal chunking."""
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
# Main K-Fold routine
# ---------------------------------------------------------------------------

def run_kfold(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Combine train + val manifests (keep test separate / unseen)
    train_items = load_manifest("train")
    val_items = load_manifest("val")
    all_items = train_items + val_items
    print(f"Total samples for K-Fold (train+val): {len(all_items)}")

    # Class distribution
    counts = Counter(it["polarity"] for it in all_items)
    print(f"Class distribution: {dict(counts)}")

    full_ds = ManifestItemDataset(all_items)
    splits = kfold_split(len(all_items), K_FOLDS, seed=42)

    fold_accuracies: List[float] = []
    fold_losses: List[float] = []

    results_lines: List[str] = [
        f"K-Fold Cross-Validation  (K={K_FOLDS}, epochs={args.epochs})\n",
        "=" * 60,
    ]

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{K_FOLDS}")
        print(f"{'='*60}")

        train_items_fold = [all_items[i] for i in train_idx]
        val_items_fold = [all_items[i] for i in val_idx]

        train_ds = Subset(full_ds, train_idx.tolist())
        val_ds = Subset(full_ds, val_idx.tolist())

        # Weighted sampler for class imbalance
        weights_dict = build_class_weights_from_items(train_items_fold)
        sampler = make_weighted_sampler(train_items_fold, weights_dict)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collate_pad,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_pad,
            num_workers=0,
        )

        # Fresh model per fold
        n_classes = len(POLARITY_TO_ID)
        model = CRNN(
            n_mels=80,
            n_classes=n_classes,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        ).to(device)

        weight_tensor = torch.tensor(
            [weights_dict.get(c, 1.0) for c in ("negative", "neutral", "positive")],
            dtype=torch.float,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

        checkpoint_path = MODELS_DIR / f"fold{fold}_model.pt"
        best_val_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(
                f"  epoch {epoch:3d}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)
            scheduler.step()

        fold_accuracies.append(best_val_acc)
        fold_losses.append(best_val_loss)
        line = (
            f"Fold {fold}: best_val_acc={best_val_acc:.4f}  "
            f"best_val_loss={best_val_loss:.4f}  "
            f"checkpoint={checkpoint_path.name}"
        )
        print(f"\n  ✓ {line}")
        results_lines.append(line)

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    summary = (
        f"\nMean Accuracy : {mean_acc:.4f}\n"
        f"Std  Accuracy : {std_acc:.4f}\n"
        f"Per-fold accs : {[round(a, 4) for a in fold_accuracies]}"
    )
    print("\n" + "=" * 60)
    print(summary)
    print("=" * 60)
    results_lines.append(summary)

    # Persist results
    results_path = MODELS_DIR / "kfold_results.txt"
    results_path.write_text("\n".join(results_lines), encoding="utf-8")
    print(f"\nResults saved → {results_path}")
    print(f"Fold models   → {MODELS_DIR}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Fold cross-validation training")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int,   default=192)
    parser.add_argument("--dropout",     type=float, default=0.3)
    run_kfold(parser.parse_args())


__all__ = ["run_kfold"]
