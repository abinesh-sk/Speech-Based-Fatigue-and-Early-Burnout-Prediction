from pathlib import Path
from typing import Iterable, Tuple

import resampy
import soundfile as sf

from .config import RAW_DATA_DIR, SAMPLE_RATE
from .preprocess import dataset_audio_paths


def _resample_file_inplace(path: Path, target_sr: int = SAMPLE_RATE) -> Tuple[bool, str]:
    """Resample to mono/target_sr in place. Returns (changed, message)."""
    try:
        audio, sr = sf.read(path)
    except Exception as exc:  # pragma: no cover - simple safeguard
        return False, f"read_failed:{exc}"

    converted_to_mono = audio.ndim > 1
    if converted_to_mono:
        audio = audio.mean(axis=1)

    needs_resample = sr != target_sr
    if needs_resample:
        audio = resampy.resample(audio, sr, target_sr)
        sr = target_sr

    # Skip rewrite if nothing changed
    if not needs_resample and not converted_to_mono:
        return False, "unchanged"

    tmp_path = path.with_name(path.stem + "_tmp" + path.suffix)
    sf.write(tmp_path, audio, sr)
    tmp_path.replace(path)
    return True, "updated"


def resample_dataset_inplace(dataset_name: str, target_sr: int = SAMPLE_RATE, extensions: Iterable[str] = (".wav",)) -> Tuple[int, int]:
    """Resample all files in data/raw/<dataset_name>/ to target_sr mono.

    Returns (updated_count, skipped_count).
    """
    dataset_root = RAW_DATA_DIR / dataset_name
    if not dataset_root.exists():
        return 0, 0

    change_flags = [
        _resample_file_inplace(path, target_sr=target_sr)[0]
        for path in dataset_audio_paths(dataset_name, extensions=extensions)
    ]
    updated = sum(change_flags)
    skipped = len(change_flags) - updated
    return updated, skipped


__all__ = ["resample_dataset_inplace"]
