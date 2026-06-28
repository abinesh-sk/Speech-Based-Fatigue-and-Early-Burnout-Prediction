"""
VoiceAnalysis DL+GRU – Export Features to CSV
===============================================
Reads the per-split manifest JSON files and the corresponding .npy
log-mel spectrogram arrays, then writes a single flat CSV where:

  • Each row  = one audio clip
  • Columns   = split, dataset, label, polarity, feature_path,
                mel_0_0, mel_0_1, …  (flattened n_mels × time_frames)

NOTE: Log-mel spectrograms are 2-D (n_mels × time_frames).
      We flatten them to 1-D for the CSV so your faculty can
      apply regression directly without reshaping.
      Column names follow the pattern  mel_{mel_bin}_{time_frame}.

Output: VoiceAnalysis DL+GRU/features_export.csv
"""

import csv
import json
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
FEATURES_DIR   = PROCESSED_DIR / "features"
OUT_CSV        = BASE_DIR / "features_export.csv"

SPLITS = ["train", "val", "test"]

# ── Collect all manifest entries ───────────────────────────────────────────────
all_entries = []
for split in SPLITS:
    manifest_path = FEATURES_DIR / split / "manifest.json"
    if not manifest_path.exists():
        print(f"  [warn] Manifest not found: {manifest_path}")
        continue
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)
    for e in entries:
        e["split"] = split
    all_entries.extend(entries)

if not all_entries:
    raise FileNotFoundError("No manifest entries found. Run extract_features.py first.")

print(f"[info] Total clips found: {len(all_entries):,}")

# ── Probe first .npy to get feature shape ─────────────────────────────────────
first_npy = PROCESSED_DIR / all_entries[0]["feature_path"]
sample = np.load(first_npy)   # shape: (n_mels, time_frames)
n_mels, n_frames = sample.shape
n_features = n_mels * n_frames
print(f"[info] Feature shape per clip: ({n_mels} mels × {n_frames} frames) = {n_features} values")

# ── Build column names ─────────────────────────────────────────────────────────
feat_cols = [f"mel_{m}_{t}" for m in range(n_mels) for t in range(n_frames)]
META_COLS = ["split", "dataset", "label", "polarity", "feature_path"]
HEADER    = META_COLS + feat_cols

# ── Write CSV ─────────────────────────────────────────────────────────────────
rows_written = 0
skipped      = 0

with open(OUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(HEADER)

    for entry in all_entries:
        npy_path = PROCESSED_DIR / entry["feature_path"]
        if not npy_path.exists():
            print(f"  [warn] Missing feature file: {npy_path}, skipping.")
            skipped += 1
            continue

        arr = np.load(npy_path).flatten().tolist()   # (n_mels × n_frames,)
        row = [
            entry.get("split", ""),
            entry.get("dataset", ""),
            entry.get("label", ""),
            entry.get("polarity", ""),
            entry.get("feature_path", ""),
        ] + arr
        writer.writerow(row)
        rows_written += 1

print(f"\n✅  DL+GRU – CSV export complete!")
print(f"   Rows written   : {rows_written:,}")
print(f"   Clips skipped  : {skipped}")
print(f"   Output file    : {OUT_CSV}")
