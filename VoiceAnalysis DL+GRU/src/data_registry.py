from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import RAW_DATA_DIR


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    url: str
    approx_files: Optional[int]
    notes: str

    @property
    def local_path(self) -> Path:
        return RAW_DATA_DIR / self.name


DATASETS = [
    DatasetInfo("RAVDESS", "https://zenodo.org/record/1188976", 1440, "Speech subset; acted emotions; clean; 48 kHz."),
    DatasetInfo("CREMA-D", "https://github.com/CheyneyComputerScience/CREMA-D", 7442, "Crowd-rated emotions; 16 kHz."),
    DatasetInfo("EMO-DB", "https://www.emodb.bilderbar.info/download/", 535, "German; clear emotions; 16 kHz."),
    DatasetInfo("TESS", "https://tspace.library.utoronto.ca/handle/1807/24487", 2800, "Seven emotions; 24 kHz."),
    DatasetInfo("IEMOCAP", "https://sail.usc.edu/iemocap/", 12_000, "Scripted + improv; registration required."),
    DatasetInfo("MSP-Podcast", "https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html", None, "Natural podcast; large; use subset."),
    DatasetInfo("SUSAS", "https://catalog.ldc.upenn.edu/LDC99S78", 9000, "Stress speech; licensed."),
    DatasetInfo("DAIC-WOZ", "http://dcapswoz.ict.usc.edu/", 189, "Clinical interviews; segment utterances."),
    DatasetInfo("VCTK", "https://datashare.ed.ac.uk/handle/10283/2651", 44_000, "Speaker baseline/embeddings."),
    DatasetInfo("LibriSpeech-clean-100", "http://www.openslr.org/12", 28_000, "Optional speaker baseline."),
]
