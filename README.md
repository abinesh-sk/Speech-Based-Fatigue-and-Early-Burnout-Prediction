<div align="center">

# 🎙️ Automated Voice-Based Analysis for Emotion Detection and Early Mental Fatigue Prediction

### A Two-Stage Deep Learning Pipeline for Annotation-Free Fatigue & Burnout Detection from Speech

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Librosa](https://img.shields.io/badge/Librosa-0.10%2B-green?style=for-the-badge)](https://librosa.org)

</div>

---

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

![End-to-end pipeline architecture of the proposed fatigue detection system](System_Architecture_Diagram.png)

> The pipeline takes raw `.wav` speech files as input, passes them through **Stage 1** (Emotion Recognition via CRNN-BiGRU) and **Stage 2** (Fatigue Prediction via BiGRU-Attention), and outputs a fatigue severity classification: **Low / Moderate / High Fatigue**.

---

## 🔬 Stage 1 — Emotion Classifier (CRNN-BiGRU)

![CRNN-BiGRU architecture used for Stage-1 speech emotion recognition](CRNN-BiGRU%20architecture%20used%20for%20Stage-1%20speech%20emotion%20recognition..png)

The Stage 1 model is a **Convolutional Recurrent Neural Network** with a Bidirectional GRU that classifies each speech utterance into one of three emotion polarities: **Negative**, **Neutral**, or **Positive**.

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

![Arousal & Fatigue Score Computation Flow — Stage 2 Formula Chain](Arousal%20%26%20Fatigue%20Score%20Computation%20Flow%20%28Stage%202%20Formula%20Chain%29.png)

The composite arousal score fuses two streams — acoustic features and emotion probabilities from Stage 1:

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

### Temporal Model (BiGRU-Attention)

![Temporal window construction and BiGRU-Attention model used for fatigue prediction](Temporal%20window%20construction%20and%20BiGRU-Attention%20model%20used%20for%20fatigue%20prediction..png)

A **10-utterance sliding window** per speaker feeds into a Bidirectional GRU with Scaled Dot-Product Attention to model the cumulative, gradual nature of burnout progression.

- **Sliding Window**: 10 consecutive utterances, stride = 5
- **Feature Vector per Timestep**: Emotion probabilities + acoustic features + arousal score + trend slope
- **Model**: BiGRU → Scaled Dot-Product Attention → Context Vector → FC → Softmax (3 classes)
- **Cross-Validation**: 5-fold stratified

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

---

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

<div align="center">

⭐ Star this repository if you found it useful!

</div>
