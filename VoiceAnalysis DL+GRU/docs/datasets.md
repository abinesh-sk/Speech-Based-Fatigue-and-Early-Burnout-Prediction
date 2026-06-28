# Dataset Download Guide (Local-Friendly)

Each dataset should be placed under `data/raw/<dataset_name>/`. Counts are approximate. Check licenses/terms before use.

| Dataset | Files / Size | URL | Notes |
| --- | --- | --- | --- |
| RAVDESS | ~2,452 files, ~24 GB (48 kHz) | https://zenodo.org/record/1188976 | Acted emotions; clean. Downsample to 16 kHz if desired. |
| CREMA-D | ~7,442 files, ~7 GB (16 kHz) | https://github.com/CheyneyComputerScience/CREMA-D | Diverse speakers; crowd-rated emotions. |
| EMO-DB | 535 files, ~100 MB (16 kHz) | https://www.emodb.bilderbar.info/download/ | German; very clear emotions. |
| TESS | ~2,800 files, ~1.5 GB (24 kHz) | https://tspace.library.utoronto.ca/handle/1807/24487 | Seven emotions; older female speakers. |
| IEMOCAP | ~12k utterances, ~12–14 GB (16 kHz) | https://sail.usc.edu/iemocap/ | Registration required; mix of scripted/improv. |
| MSP-Podcast (subset) | Very large; pull a filtered 20–50k clip subset | https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html | Natural podcast speech; weak labels; filter for balanced subset. |
| SUSAS | ~9k utterances | https://catalog.ldc.upenn.edu/LDC99S78 | Stress speech; license via LDC. |
| DAIC-WOZ | 189 sessions (long) | http://dcapswoz.ict.usc.edu/ | Clinical interviews; segment into utterances. |
| VCTK | ~44k clips, ~10–15 GB (44.1 kHz) | https://datashare.ed.ac.uk/handle/10283/2651 | Speaker baselines/embeddings. |
| LibriSpeech (clean-100) | ~28k clips, 6.3 GB (16 kHz) | http://www.openslr.org/12 | Optional for speaker normalization pretraining. |
| COSWARA (optional) | voice/breath/cough, GBs | https://coswara.iisc.ac.in/ | Health-related; use cautiously for vocal fatigue cues. |

**Download tips (local constraints):**
- Start small: RAVDESS + CREMA-D + EMO-DB + TESS (~13k clips) to prototype.
- Add IEMOCAP next; then a filtered MSP-Podcast slice if storage allows.
- Keep archives; extract to `data/raw/<dataset>`; downsample to 16 kHz mono for consistency.
