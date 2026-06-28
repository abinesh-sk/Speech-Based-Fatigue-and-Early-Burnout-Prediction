"""
Dataset Statistics Generator
=============================
Reads train.csv, val.csv, test.csv from the metadata directory and generates
a comprehensive dataset_statistics.txt report covering:
  - Per-split sample counts
  - Per-class (polarity) distribution per split
  - Per-dataset source distribution per split
  - Class imbalance ratios
  - Overall totals
  - Imbalance handling strategy notes

Run:
    python dataset_stats.py
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

# Resolve paths relative to this script's location
PROJECT_ROOT = Path(__file__).resolve().parent
METADATA_DIR = PROJECT_ROOT / "data" / "processed" / "metadata"
OUTPUT_PATH = PROJECT_ROOT / "dataset_statistics.txt"


def load_split(split: str) -> List[Dict[str, str]]:
    """Load a CSV split into a list of row dicts."""
    path = METADATA_DIR / f"{split}.csv"
    if not path.exists():
        print(f"  [WARN] {path} not found, skipping.")
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def polarity_counts(rows: List[Dict]) -> Counter:
    return Counter(r["polarity"] for r in rows)


def label_counts(rows: List[Dict]) -> Counter:
    return Counter(r["label"] for r in rows)


def dataset_counts(rows: List[Dict]) -> Counter:
    return Counter(r["dataset"] for r in rows)


def imbalance_ratio(counts: Counter) -> float:
    """Max / Min class count ratio (higher = more imbalanced)."""
    if not counts:
        return 0.0
    vals = list(counts.values())
    return max(vals) / max(min(vals), 1)


def format_counter(counter: Counter, total: int, indent: int = 4) -> str:
    pad = " " * indent
    lines = []
    for k, v in sorted(counter.items()):
        pct = 100.0 * v / total if total else 0.0
        lines.append(f"{pad}{k:<25}  {v:>6}  ({pct:.1f}%)")
    return "\n".join(lines)


def generate_report() -> str:
    splits = ("train", "val", "test")
    split_data = {s: load_split(s) for s in splits}

    all_rows = []
    for rows in split_data.values():
        all_rows.extend(rows)

    lines: List[str] = []

    lines.append("=" * 70)
    lines.append("  DATASET STATISTICS REPORT — Speech Emotion Recognition")
    lines.append("=" * 70)
    lines.append("")

    # ── Per-split summary ──────────────────────────────────────────────────
    lines.append("─" * 70)
    lines.append("  1. PER-SPLIT SAMPLE COUNTS")
    lines.append("─" * 70)
    lines.append(f"  {'Split':<10}  {'Samples':>8}")
    for s in splits:
        n = len(split_data[s])
        lines.append(f"  {s:<10}  {n:>8}")
    lines.append(f"  {'TOTAL':<10}  {len(all_rows):>8}")
    lines.append("")

    # ── Class (polarity) distribution ─────────────────────────────────────
    lines.append("─" * 70)
    lines.append("  2. CLASS (POLARITY) DISTRIBUTION")
    lines.append("─" * 70)
    for s in splits:
        rows = split_data[s]
        if not rows:
            continue
        counts = polarity_counts(rows)
        ratio = imbalance_ratio(counts)
        lines.append(f"\n  [{s.upper()} — {len(rows)} samples]  imbalance ratio = {ratio:.2f}x")
        lines.append(f"  {'Class':<25}  {'Count':>6}  Pct")
        lines.append(format_counter(counts, len(rows)))

    # Overall polarity
    all_pol = polarity_counts(all_rows)
    lines.append(f"\n  [OVERALL — {len(all_rows)} samples]")
    lines.append(f"  {'Class':<25}  {'Count':>6}  Pct")
    lines.append(format_counter(all_pol, len(all_rows)))
    lines.append(f"\n  Overall imbalance ratio: {imbalance_ratio(all_pol):.2f}x")
    lines.append("")

    # ── Fine-grained emotion label distribution ────────────────────────────
    lines.append("─" * 70)
    lines.append("  3. FINE-GRAINED EMOTION LABEL DISTRIBUTION (OVERALL)")
    lines.append("─" * 70)
    all_labels = label_counts(all_rows)
    lines.append(f"  {'Label':<25}  {'Count':>6}  Pct")
    lines.append(format_counter(all_labels, len(all_rows)))
    lines.append("")

    # ── Source dataset distribution ────────────────────────────────────────
    lines.append("─" * 70)
    lines.append("  4. SOURCE DATASET DISTRIBUTION")
    lines.append("─" * 70)
    for s in splits:
        rows = split_data[s]
        if not rows:
            continue
        ds_counts = dataset_counts(rows)
        lines.append(f"\n  [{s.upper()}]")
        lines.append(f"  {'Dataset':<25}  {'Count':>6}  Pct")
        lines.append(format_counter(ds_counts, len(rows)))

    all_ds = dataset_counts(all_rows)
    lines.append(f"\n  [OVERALL]")
    lines.append(f"  {'Dataset':<25}  {'Count':>6}  Pct")
    lines.append(format_counter(all_ds, len(all_rows)))
    lines.append("")

    # ── Class imbalance handling notes ────────────────────────────────────
    lines.append("─" * 70)
    lines.append("  5. CLASS IMBALANCE HANDLING STRATEGIES")
    lines.append("─" * 70)
    lines.append("""
  Two complementary strategies are applied during model training:

  a) WeightedRandomSampler
     ─────────────────────
     Each training sample is assigned a weight inversely proportional to
     its class frequency.  The sampler draws batches with replacement so
     that all classes are represented roughly equally in every mini-batch.
     Formula: w_c = total_samples / (n_classes × count_c)

  b) Class-Weighted Cross-Entropy Loss
     ────────────────────────────────
     The same inverse-frequency weights are passed to nn.CrossEntropyLoss
     so that misclassifying minority-class samples incurs a larger penalty,
     biasing the gradient updates toward harder-to-learn classes.

  Together these techniques counteract over-representation of the majority
  class throughout both sampling and the gradient signal.
""")

    lines.append("=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_report()
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved → {OUTPUT_PATH}")
