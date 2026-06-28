from pathlib import Path
from typing import Optional

import soundfile as sf

from .config import SAMPLE_RATE


def load_audio(path: Path, target_sr: int = SAMPLE_RATE) -> Optional[tuple]:
    """Load audio and resample if needed. Returns (waveform, sr) or None on failure."""
    try:
        audio, sr = sf.read(path)
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        # Lazy import to avoid heavy dep if unused
        import resampy

        audio = resampy.resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


__all__ = ["load_audio"]
