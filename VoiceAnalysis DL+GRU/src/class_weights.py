from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict

from .config import METADATA_DIR


def compute_class_weights(csv_path: Path = METADATA_DIR / "train.csv", label_key: str = "polarity") -> Dict[str, float]:
    counts: Counter[str] = Counter()
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row[label_key]] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    weights = {lbl: total / (len(counts) * cnt) for lbl, cnt in counts.items()}
    return weights


if __name__ == "__main__":
    weights = compute_class_weights()
    print("Class weights (train split):")
    for k, v in weights.items():
        print(f"  {k}: {v:.4f}")


__all__ = ["compute_class_weights"]
