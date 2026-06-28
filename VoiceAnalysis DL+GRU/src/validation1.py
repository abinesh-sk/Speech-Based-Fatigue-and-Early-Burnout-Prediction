from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PROCESSED_DATA_DIR, METADATA_DIR
from .data_loader import MelDataset, collate_pad, POLARITY_TO_ID
from .model import CRNN

ID_TO_POLARITY: Dict[int, str] = {v: k for k, v in POLARITY_TO_ID.items()}


def load_manifest(split: str) -> Path:
    return PROCESSED_DATA_DIR / "features" / split / "manifest.json"


def collect_preds(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    assert preds.shape == labels.shape
    classes = sorted(np.unique(np.concatenate([preds, labels])))
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for p, t in zip(preds, labels):
        confusion[t, p] += 1

    per_class_prec = []
    per_class_rec = []
    per_class_f1 = []
    for i, _c in enumerate(classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        per_class_prec.append(prec)
        per_class_rec.append(rec)
        per_class_f1.append(f1)

    acc = float((preds == labels).mean())
    macro_prec = float(np.mean(per_class_prec))
    macro_rec = float(np.mean(per_class_rec))
    macro_f1 = float(np.mean(per_class_f1))
    return {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
    }


def run_eval(
    checkpoint: str,
    split: str = "test",
    batch_size: int = 16,
    num_workers: int = 0,
    hidden_size: int = 192,
    dropout: float = 0.3,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = load_manifest(split)
    dataset = MelDataset(manifest_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_pad)

    model = CRNN(n_mels=80, n_classes=len(POLARITY_TO_ID), hidden_size=hidden_size, dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    preds, labels = collect_preds(model, loader, device)
    metrics = compute_metrics(preds, labels)

    print(json.dumps(metrics, indent=2))

    # Optional: print per-class counts
    counts = {ID_TO_POLARITY[i]: int((labels == i).sum()) for i in np.unique(labels)}
    print("per_class_counts:", counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="crnn_best.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()
    run_eval(
        checkpoint=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()


__all__ = ["run_eval"]
