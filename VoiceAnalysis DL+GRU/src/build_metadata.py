from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .config import METADATA_DIR, RAW_DATA_DIR
from .preprocess import dataset_audio_paths

# Emotion to polarity mapping
POSITIVE = {"happy", "hap", "happ", "surprised", "surprise", "sur", "ps", "pleasant", "pleasant_surprise"}
NEUTRAL = {"neutral", "neu", "calm", "cal"}
NEGATIVE = {"sad", "angry", "fear", "disgust", "dis", "fea", "ang", "sadness"}


def _polarity_from_label(label: str) -> Optional[str]:
    lab = label.lower()
    if lab in POSITIVE:
        return "positive"
    if lab in NEUTRAL:
        return "neutral"
    if lab in NEGATIVE:
        return "negative"
    return None


def _ravdess_label(path: Path) -> Optional[str]:
    """RAVDESS filename pattern: MM-VC-EM-IM-ST-RT-AC.wav; use token 3."""
    parts = path.stem.split("-")
    if len(parts) < 3:
        return None
    em_code = parts[2]
    code_map: Dict[str, str] = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprised",
    }
    return code_map.get(em_code)


def _cremad_label(path: Path) -> Optional[str]:
    """CREMA-D pattern: <actorID>_<something>_<EMO>_<...>.wav; token 3 holds emotion code."""
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    emo_code = parts[2].lower()
    code_map = {
        "ang": "angry",
        "dis": "disgust",
        "fea": "fear",
        "hap": "happy",
        "neu": "neutral",
        "sad": "sad",
    }
    return code_map.get(emo_code)


def _tess_label(path: Path) -> Optional[str]:
    """TESS pattern: OAF_back_angry.wav -> emotion is last token."""
    parts = path.stem.split("_")
    if len(parts) < 2:
        return None
    return parts[-1]


def _label_for(dataset: str, path: Path) -> Optional[str]:
    if dataset == "RAVDESS":
        return _ravdess_label(path)
    if dataset == "CREMA-D":
        return _cremad_label(path)
    if dataset == "TESS":
        return _tess_label(path)
    return None


def iter_metadata_rows(datasets: Iterable[str]) -> Iterator[Dict[str, str]]:
    """Yield metadata rows for present datasets."""
    for ds in datasets:
        for audio_path in dataset_audio_paths(ds):
            label = _label_for(ds, audio_path)
            if label is None:
                continue
            polarity = _polarity_from_label(label)
            if polarity is None:
                continue
            yield {
                "dataset": ds,
                "path": str(audio_path.relative_to(RAW_DATA_DIR)),
                "label": label,
                "polarity": polarity,
            }


def build_metadata_csv(datasets: Iterable[str], output: Path = METADATA_DIR / "metadata.csv") -> Path:
    rows = list(iter_metadata_rows(datasets))
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = ["dataset", "path", "label", "polarity"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def available_datasets() -> List[str]:
    return [p.name for p in RAW_DATA_DIR.iterdir() if p.is_dir()]


if __name__ == "__main__":
    out_path = build_metadata_csv(available_datasets())
    print(f"Wrote metadata: {out_path}")


__all__ = [
    "iter_metadata_rows",
    "build_metadata_csv",
    "available_datasets",
]
