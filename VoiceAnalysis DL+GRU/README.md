# Voice Analysis for Emotion and Fatigue (Local)

## Scope
End-to-end speech pipeline (local) for emotion polarity (positive/neutral/negative) and fatigue/burnout trend detection using Mel-spectrogram + acoustic features, multitask CRNN, and temporal trend modeling.

## Datasets to fetch (links and details in [docs/datasets.md](docs/datasets.md))
- Core emotion (acted, clean): RAVDESS, CREMA-D, EMO-DB, TESS.
- Natural/large emotion: IEMOCAP, MSP-Podcast subset (filter to manageable size).
- Stress/fatigue proxies: SUSAS, DAIC-WOZ (segment interviews), optional COSWARA/voice-fatigue if accessible.
- Speaker baselines: VCTK (or LibriSpeech-clean-100) for per-speaker normalization/embeddings.

## Hardware note
Designed to run locally on 16 GB RAM + CPU/Integrated GPU. Use small batch sizes, compact models (MobileNet-ish CNN + GRU), and start with the small acted sets before adding bigger corpora.

## Project layout
- data/raw/ — place downloaded archives or extracted audio here (per-dataset subfolders).
- data/processed/ — cached features/metadata.
- docs/datasets.md — download links, counts, license notes.
- src/ — pipeline code (preprocessing, feature extraction, metadata utilities).

## Quick next steps
1) Download the listed datasets into data/raw/ (see docs/datasets.md).
2) Populate a metadata CSV later (path, speaker, split, labels). Generators in src/preprocess.py are ready for file walking.
3) Implement feature extraction (Mel, MFCC, pitch) and model training scripts once data is in place.
