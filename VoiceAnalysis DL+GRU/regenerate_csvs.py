"""Quick script to regenerate acoustic feature CSVs with mel summary stats.
Does NOT touch .npy files (safe to run while K-Fold training is in progress).
"""
import csv
import sys
from pathlib import Path

import librosa
import numpy as np

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ACOUSTIC_FEATURES_DIR, METADATA_DIR, PROCESSED_DATA_DIR, SAMPLE_RATE, N_MELS, HOP_LENGTH_MS, WIN_LENGTH_MS
from src.audio_utils import load_audio
from src.extract_features import (
    loudness_norm, trim_pad, compute_log_mel, pad_mel_to_fixed,
    extract_pitch_features, extract_energy_features, extract_zcr_features,
    extract_centroid_features, extract_rolloff_features, MAX_SAMPLES,
)

def process_all():
    all_rows = []
    for split in ("train", "val", "test"):
        meta_path = METADATA_DIR / f"{split}.csv"
        count = 0
        with meta_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_path = Path("data/raw") / row["path"]
                loaded = load_audio(audio_path)
                if loaded is None:
                    continue
                audio, sr = loaded
                audio = loudness_norm(audio)
                audio_duration = len(audio) / sr
                audio = trim_pad(audio, MAX_SAMPLES)

                log_mel = compute_log_mel(audio, sr)
                log_mel_fixed = pad_mel_to_fixed(log_mel)

                pitch = extract_pitch_features(audio, sr)
                energy = extract_energy_features(audio)
                zcr = extract_zcr_features(audio)
                centroid = extract_centroid_features(audio, sr)
                rolloff = extract_rolloff_features(audio, sr)

                csv_row = {
                    "audio_path": str(audio_path),
                    "label": row["label"],
                    "polarity": row["polarity"],
                    "dataset": row["dataset"],
                    "split": split,
                    "audio_duration": round(audio_duration, 4),
                    "sample_rate": sr,
                    **pitch, **energy, **zcr, **centroid, **rolloff,
                    "mel_mean": round(float(np.mean(log_mel_fixed)), 4),
                    "mel_std": round(float(np.std(log_mel_fixed)), 4),
                    "mel_min": round(float(np.min(log_mel_fixed)), 4),
                    "mel_max": round(float(np.max(log_mel_fixed)), 4),
                }
                all_rows.append(csv_row)
                count += 1
        print(f"[{split}] processed {count} samples")

    # Clean old CSVs
    for old in ACOUSTIC_FEATURES_DIR.glob("acoustic_features_part*.csv"):
        old.unlink()

    # Write chunked CSVs
    chunk_size = 3000
    fieldnames = list(all_rows[0].keys())
    for part_idx, start in enumerate(range(0, len(all_rows), chunk_size), start=1):
        chunk = all_rows[start:start + chunk_size]
        path = ACOUSTIC_FEATURES_DIR / f"acoustic_features_part{part_idx}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(chunk)
        print(f"  Wrote {path.name} ({len(chunk)} rows)")

    print(f"\nTotal: {len(all_rows)} samples → {ACOUSTIC_FEATURES_DIR}")

if __name__ == "__main__":
    process_all()
