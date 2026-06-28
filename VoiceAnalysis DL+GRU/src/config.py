from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROCESSED_DATA_DIR / "metadata"
MODELS_DIR = PROJECT_ROOT / "models"
ACOUSTIC_FEATURES_DIR = PROCESSED_DATA_DIR / "acoustic_features"

# Audio defaults
SAMPLE_RATE = 16_000
N_MELS = 80
WIN_LENGTH_MS = 25
HOP_LENGTH_MS = 10

# Create required directories at import time (idempotent)
for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR, MODELS_DIR, ACOUSTIC_FEATURES_DIR):
    path.mkdir(parents=True, exist_ok=True)
