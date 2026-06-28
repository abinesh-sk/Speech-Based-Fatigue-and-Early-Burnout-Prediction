from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from .config import (
    ACOUSTIC_FEATURES_DIR,
    METADATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_RATE,
    N_MELS,
    WIN_LENGTH_MS,
    HOP_LENGTH_MS,
)
from .audio_utils import load_audio

WIN_LENGTH = int(SAMPLE_RATE * WIN_LENGTH_MS / 1000)
HOP_LENGTH = int(SAMPLE_RATE * HOP_LENGTH_MS / 1000)

# Fixed Mel frame count for consistent .npy shapes (3 s @ 16 kHz, hop=160)
MAX_DURATION_S: float = 3.0
MAX_SAMPLES: int = int(MAX_DURATION_S * SAMPLE_RATE)

# Mel spectrogram dimensions after trim/pad
_MAX_FRAMES = 1 + MAX_SAMPLES // HOP_LENGTH  # ≈ 301
N_MEL_COLS = N_MELS * _MAX_FRAMES            # total flattened mel columns


# ---------------------------------------------------------------------------
# Audio preprocessing helpers
# ---------------------------------------------------------------------------

def loudness_norm(audio: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2) + eps)
    target = 0.1
    return audio * (target / max(rms, eps))


def trim_pad(audio: np.ndarray, max_samples: int = MAX_SAMPLES) -> np.ndarray:
    if len(audio) > max_samples:
        return audio[:max_samples]
    elif len(audio) < max_samples:
        return np.pad(audio, (0, max_samples - len(audio)))
    return audio


# ---------------------------------------------------------------------------
# Log Mel Spectrogram
# ---------------------------------------------------------------------------

def compute_log_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=2048,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        center=True,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def pad_mel_to_fixed(log_mel: np.ndarray) -> np.ndarray:
    """Pad/trim mel spectrogram along the time axis so it has exactly _MAX_FRAMES columns."""
    n_mels, t = log_mel.shape
    if t < _MAX_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, _MAX_FRAMES - t)))
    elif t > _MAX_FRAMES:
        log_mel = log_mel[:, :_MAX_FRAMES]
    return log_mel


# ---------------------------------------------------------------------------
# Acoustic feature extractors
# ---------------------------------------------------------------------------

def extract_pitch_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Pitch stats using librosa.yin (fast F0 estimator)."""
    try:
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=HOP_LENGTH,
        )
        voiced = f0[f0 > 0]
        if len(voiced) == 0:
            return {"mean_pitch": 0.0, "std_pitch": 0.0, "min_pitch": 0.0, "max_pitch": 0.0}
        return {
            "mean_pitch": float(np.mean(voiced)),
            "std_pitch": float(np.std(voiced)),
            "min_pitch": float(np.min(voiced)),
            "max_pitch": float(np.max(voiced)),
        }
    except Exception:
        return {"mean_pitch": 0.0, "std_pitch": 0.0, "min_pitch": 0.0, "max_pitch": 0.0}


def extract_energy_features(audio: np.ndarray) -> Dict[str, float]:
    """RMS energy stats."""
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    return {
        "mean_energy": float(np.mean(rms)),
        "std_energy": float(np.std(rms)),
        "max_energy": float(np.max(rms)),
    }


def extract_zcr_features(audio: np.ndarray) -> Dict[str, float]:
    """Zero Crossing Rate stats."""
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=HOP_LENGTH)[0]
    return {
        "mean_zcr": float(np.mean(zcr)),
        "std_zcr": float(np.std(zcr)),
    }


def extract_centroid_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Spectral centroid stats."""
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
    return {
        "mean_centroid": float(np.mean(centroid)),
        "std_centroid": float(np.std(centroid)),
    }


def extract_rolloff_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Spectral rolloff stats."""
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
    return {
        "mean_rolloff": float(np.mean(rolloff)),
        "std_rolloff": float(np.std(rolloff)),
    }


# ---------------------------------------------------------------------------
# Single-file feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(
    audio_path: Path,
    label: str,
    polarity: str,
    dataset: str,
    split: str,
) -> Optional[Tuple[np.ndarray, Dict]]:
    """Return (log_mel_array, row_dict) or None on failure."""
    loaded = load_audio(audio_path)
    if loaded is None:
        return None
    audio, sr = loaded
    audio = loudness_norm(audio)
    audio_duration = len(audio) / sr
    audio = trim_pad(audio, MAX_SAMPLES)

    # Mel spectrogram (.npy feature)
    log_mel = compute_log_mel(audio, sr)            # (N_MELS, T)
    log_mel_fixed = pad_mel_to_fixed(log_mel)       # (N_MELS, _MAX_FRAMES)

    # Acoustic descriptors
    pitch_feats = extract_pitch_features(audio, sr)
    energy_feats = extract_energy_features(audio)
    zcr_feats = extract_zcr_features(audio)
    centroid_feats = extract_centroid_features(audio, sr)
    rolloff_feats = extract_rolloff_features(audio, sr)

    # Build CSV row
    row: Dict = {
        "audio_path": str(audio_path),
        "label": label,
        "polarity": polarity,
        "dataset": dataset,
        "split": split,
        "audio_duration": round(audio_duration, 4),
        "sample_rate": sr,
        **pitch_feats,
        **energy_feats,
        **zcr_feats,
        **centroid_feats,
        **rolloff_feats,
    }

    # Mel summary statistics (keeping CSV compact for faculty review)
    row["mel_mean"] = round(float(np.mean(log_mel_fixed)), 4)
    row["mel_std"] = round(float(np.std(log_mel_fixed)), 4)
    row["mel_min"] = round(float(np.min(log_mel_fixed)), 4)
    row["mel_max"] = round(float(np.max(log_mel_fixed)), 4)

    return log_mel, row


# ---------------------------------------------------------------------------
# Per-split processing (saves .npy + collects CSV rows)
# ---------------------------------------------------------------------------

def process_split(split: str) -> Tuple[Path, Path, List[Dict]]:
    """Process one split. Returns (out_dir, manifest_path, list_of_csv_rows)."""
    meta_path = METADATA_DIR / f"{split}.csv"
    out_dir = PROCESSED_DATA_DIR / "features" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Delete stale .npy files
    for old_npy in out_dir.glob("*.npy"):
        old_npy.unlink()

    manifest: List[Dict[str, str]] = []
    csv_rows: List[Dict] = []

    with meta_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = Path("data/raw") / row["path"]
            result = extract_all_features(
                audio_path,
                label=row["label"],
                polarity=row["polarity"],
                dataset=row["dataset"],
                split=split,
            )
            if result is None:
                continue
            log_mel, csv_row = result

            # Save .npy
            feat_path = out_dir / (audio_path.stem + ".npy")
            np.save(feat_path, log_mel)

            manifest.append(
                {
                    "feature_path": str(feat_path.relative_to(PROCESSED_DATA_DIR)),
                    "label": row["label"],
                    "polarity": row["polarity"],
                    "dataset": row["dataset"],
                }
            )
            csv_rows.append(csv_row)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    print(f"[{split}] saved {len(manifest)} .npy files → {out_dir}")
    return out_dir, manifest_path, csv_rows


# ---------------------------------------------------------------------------
# CSV chunked writer
# ---------------------------------------------------------------------------

def write_csv_chunks(all_rows: List[Dict], chunk_size: int = 3000) -> List[Path]:
    """Write rows to chunked CSVs in ACOUSTIC_FEATURES_DIR."""
    if not all_rows:
        return []

    fieldnames = list(all_rows[0].keys())
    paths: List[Path] = []
    for part_idx, start in enumerate(range(0, len(all_rows), chunk_size), start=1):
        chunk = all_rows[start : start + chunk_size]
        csv_path = ACOUSTIC_FEATURES_DIR / f"acoustic_features_part{part_idx}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(chunk)
        paths.append(csv_path)
        print(f"  Wrote {csv_path.name}  ({len(chunk)} rows)")
    return paths


# ---------------------------------------------------------------------------
# Main entry-points
# ---------------------------------------------------------------------------

def run_all_splits():
    """Process train/val/test splits, save .npy files and write chunked CSVs."""
    # Clean old CSV files first
    for old_csv in ACOUSTIC_FEATURES_DIR.glob("acoustic_features_part*.csv"):
        old_csv.unlink()

    all_csv_rows: List[Dict] = []
    outputs = {}
    for split in ("train", "val", "test"):
        feat_dir, manifest, csv_rows = process_split(split)
        outputs[split] = {"features": str(feat_dir), "manifest": str(manifest)}
        all_csv_rows.extend(csv_rows)

    print(f"\nTotal samples collected: {len(all_csv_rows)}")
    csv_paths = write_csv_chunks(all_csv_rows)
    outputs["csv_files"] = [str(p) for p in csv_paths]
    return outputs


if __name__ == "__main__":
    out = run_all_splits()
    print(json.dumps({k: v for k, v in out.items() if k != "csv_files"}, indent=2))
    print("CSV parts written:", out.get("csv_files", []))


__all__ = [
    "run_all_splits",
    "process_split",
    "compute_log_mel",
    "loudness_norm",
    "extract_all_features",
]
