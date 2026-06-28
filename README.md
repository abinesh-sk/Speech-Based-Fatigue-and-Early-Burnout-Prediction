<div align="center">

# 🎙️ Automated Voice-Based Analysis for Emotion Detection and Early Mental Fatigue Prediction

### A Two-Stage Deep Learning Pipeline for Annotation-Free Fatigue & Burnout Detection from Speech

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Librosa](https://img.shields.io/badge/Librosa-0.10%2B-green?style=for-the-badge)](https://librosa.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)


</div>


## 🧠 Overview

Occupational **fatigue and burnout** are among the most widespread workplace health crises, officially recognized by the WHO as a syndrome resulting from unmanaged chronic stress. Current diagnostic methods — questionnaires (like the Maslach Burnout Inventory), clinical interviews, or physiological sensors — are either intrusive, expensive, or impractical for continuous monitoring.

This project proposes a fully **automated, annotation-free, two-stage pipeline** that detects early signs of mental fatigue and burnout **purely from speech signals** — no physiological sensors, no video, no text transcriptions required.

### 🔑 What Makes This System Unique

| Feature | Description |
|---|---|
| **Annotation-Free** | No fatigue-specific labels needed — pseudo-labels are generated automatically |
| **Speaker-Independent** | No per-speaker calibration or enrollment required |
| **Audio-Only** | Works purely on raw speech, completely non-invasive |
| **Theoretically Grounded** | Based on Russell's Circumplex Model of Affect + Maslach Burnout Inventory |
| **Reproducible** | All datasets and libraries are open-source |

---

## 🏗️ System Architecture

```
Raw Speech Audio
       │
       ▼
┌──────────────────────────────────────────────────┐
│         AUDIO PREPROCESSING                      │
│  16kHz Mono Resampling → Silence Removal →       │
│  Loudness Normalization → 3-second Padding       │
└─────────────────────┬────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   Log-Mel Spectrogram     6 Acoustic Features
   (80 mel bands)          (RMS, F0, Spectral, ZCR)
          │                       │
          ▼                       │
┌─────────────────────┐           │
│  STAGE 1            │           │
│  CRNN-BiGRU         │           │
│  Emotion Classifier │           │
│  → Negative         │           │
│  → Neutral          │           │
│  → Positive         │           │
└──────────┬──────────┘           │
           │  Class Probabilities  │
           └──────────┬────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│         STAGE 2: AROUSAL COMPUTATION            │
│  A_acoustic = f(RMS, F0, Spectral, ZCR)         │
│  A_emotion  = f(P_negative, P_neutral, P_pos)   │
│  A_final    = α·A_emotion + (1-α)·A_acoustic    │
│  Fatigue Score = g(A_final, sustained affect)   │
│  K-Means Clustering → Pseudo-Labels             │
│  [Non-Fatigued | Mild Fatigue | High Fatigue]   │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  STAGE 2: BiGRU-ATTENTION TEMPORAL MODEL        │
│  Sliding Window (10 utterances per speaker)     │
│         ↓                                        │
│  Bidirectional GRU + Scaled Dot-Product Attention│
│         ↓                                        │
│  [Non-Fatigued | Mild Fatigue | High Burnout]   │
└─────────────────────────────────────────────────┘
```

---

## 🔬 Stage 1 — Emotion Classifier (CRNN-BiGRU)

### Model Architecture

| Layer | Details |
|---|---|
| **Input** | Log-Mel Spectrogram (80 mel bands, 3-second clips) |
| **Conv Block 1** | Conv2D(1→32) + BatchNorm + ReLU + MaxPool |
| **Conv Block 2** | Conv2D(32→64) + BatchNorm + ReLU + MaxPool |
| **Conv Block 3** | Conv2D(64→128) + BatchNorm + ReLU + MaxPool |
| **BiGRU** | Bidirectional GRU, Hidden Size = 192 |
| **Dropout** | 0.3 |
| **Output** | 3 classes (Negative, Neutral, Positive) |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-3 (with ReduceLROnPlateau) |
| Batch Size | 32 |
| Epochs | 50 (Early Stopping, patience=10) |
| Class Imbalance | WeightedRandomSampler + class-weighted CrossEntropyLoss |
| Validation | 5-Fold Stratified Cross-Validation (SEED=42) |

### Emotion-to-Polarity Mapping

| Original Emotion | Polarity Label | Arousal Weight |
|---|---|---|
| Angry, Fearful, Disgusted | Negative (High-Arousal) | 0.8 |
| Sad, Calm, Bored | Negative (Low-Arousal) | 0.2 |
| Neutral | Neutral | 0.4 |
| Happy, Surprised | Positive (High-Arousal) | 0.9 |

---

## ⏱️ Stage 2 — Fatigue Detection (BiGRU-Attention)

### Arousal Score Computation

**Acoustic Arousal (A_acoustic):**
- RMS Energy (weight: 0.30)
- F0 Mean — Fundamental Frequency mean (weight: 0.25)
- F0 Variance (weight: 0.20)
- Spectral Centroid (weight: 0.15)
- Spectral Rolloff (weight: 0.05)
- Zero Crossing Rate (weight: 0.05)

**Composite Arousal Formula:**
```
A_final = α · A_emotion + (1 - α) · A_acoustic
```

### Pseudo-Label Generation (K-Means Clustering)

| Cluster | Label | Description |
|---|---|---|
| Low Fatigue Score | **Non-Fatigued** | Normal arousal, positive/neutral affect |
| Mid Fatigue Score | **Mild Fatigue** | Reduced arousal, mixed affect |
| High Fatigue Score | **High Fatigue / Burnout** | Sustained low arousal, negative affect |

---

## 📂 Datasets

| Dataset | Samples | % of Total | Description |
|---|---|---|---|
| **CREMA-D** | 7,442 | 63.7% | Crowd-sourced Emotional Multimodal Actors Dataset |
| **TESS** | 2,800 | 24.0% | Toronto Emotional Speech Set |
| **RAVDESS** | 1,440 | 12.3% | Ryerson Audio-Visual Database of Emotional Speech and Song |
| **Total** | **11,682** | 100% | |

> ⚠️ **Datasets not included** in this repo due to size. Download from:
> - [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
> - [RAVDESS](https://zenodo.org/record/1188976)
> - [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)


## 📚 Research Background

Grounded in:
1. **Russell's Circumplex Model of Affect (1980)** — valence-arousal space for emotion-to-fatigue mapping
2. **Maslach Burnout Inventory (MBI)** — three-tier fatigue severity framework
3. **CRNN-BiGRU SER literature** — Trigeorgis et al. (2016), Schuller et al. (2013)

---

## 🔭 Future Enhancements

- [ ] Real-world naturalistic speech datasets
- [ ] Per-speaker baseline calibration
- [ ] wav2vec 2.0 / HuBERT as encoder
- [ ] Real-time REST API deployment
- [ ] Multi-modal fusion (speech + facial expression)

---
