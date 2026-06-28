from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import PROCESSED_DATA_DIR

# Fixed polarity mapping; keep consistent across training/eval
POLARITY_TO_ID: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


class MelDataset(Dataset):
    def __init__(self, manifest_path: Path, max_frames: int | None = None):
        self.manifest_path = manifest_path
        with manifest_path.open() as f:
            self.items: List[Dict[str, str]] = json.load(f)
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.items[idx]
        # In manifest.json generated on Windows, feature_path contains '\\'
        # Convert any backslashes to forward slashes to make it work on Linux (Colab)
        feat_path_str = item["feature_path"].replace('\\', '/')
        feat_path = PROCESSED_DATA_DIR / Path(feat_path_str)
        mel = np.load(feat_path)
        # mel shape: (n_mels, time)
        if self.max_frames is not None:
            mel = mel[:, : self.max_frames]
        mel_tensor = torch.from_numpy(mel)  # float32
        label = POLARITY_TO_ID[item["polarity"]]
        return mel_tensor, label


def collate_pad(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    mels, labels = zip(*batch)
    # Pad to max length in batch
    max_len = max(m.shape[1] for m in mels)
    padded = []
    for m in mels:
        if m.shape[1] < max_len:
            pad_amt = max_len - m.shape[1]
            pad_tensor = torch.nn.functional.pad(m, (0, pad_amt))
            padded.append(pad_tensor)
        else:
            padded.append(m)
    stacked = torch.stack(padded)  # (B, n_mels, T)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return stacked, labels_t


__all__ = ["MelDataset", "collate_pad", "POLARITY_TO_ID"]
