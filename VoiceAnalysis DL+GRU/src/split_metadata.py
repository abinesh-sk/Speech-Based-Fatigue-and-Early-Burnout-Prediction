from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .config import METADATA_DIR


def _stratified_split(rows: List[Dict[str, str]], label_key: str, ratios: Tuple[float, float, float], seed: int = 42) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Stratified split into train/val/test by label_key."""
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    by_label: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_label.setdefault(r[label_key], []).append(r)

    rng = random.Random(seed)
    train: List[Dict[str, str]] = []
    val: List[Dict[str, str]] = []
    test: List[Dict[str, str]] = []

    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    return train, val, test


def split_metadata(
    source_csv: Path = METADATA_DIR / "metadata.csv",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Path, Path, Path]:
    rows: List[Dict[str, str]] = []
    with source_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    train, val, test = _stratified_split(rows, label_key="polarity", ratios=(train_ratio, val_ratio, test_ratio), seed=seed)

    dests = {
        "train": METADATA_DIR / "train.csv",
        "val": METADATA_DIR / "val.csv",
        "test": METADATA_DIR / "test.csv",
    }
    for split_name, split_rows in ("train", train), ("val", val), ("test", test):
        with dests[split_name].open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(split_rows)

    return dests["train"], dests["val"], dests["test"]


if __name__ == "__main__":
    out = split_metadata()
    print("Wrote splits:", out)


__all__ = ["split_metadata"]
