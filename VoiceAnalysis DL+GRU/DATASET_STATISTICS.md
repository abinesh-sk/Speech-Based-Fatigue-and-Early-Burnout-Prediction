# Dataset Statistics Report

## Summary

Based on analysis of the manifest files in `data/processed/features/`, here is the complete breakdown of your VoiceAnalysis dataset:

---

## Overall Statistics

**Total Samples: 11,682**

### Split Distribution

| Split | Samples | Percentage |
|-------|---------|------------|
| **Train** | 9,345 | 80.0% |
| **Validation** | 1,167 | 10.0% |
| **Test** | 1,170 | 10.0% |

---

## Class Distribution by Split

| Class | Train | Validation | Test | **Total** |
|-------|-------|------------|------|-----------|
| **Negative** | 5,969 | 743 | 746 | **7,458** |
| **Neutral** | 1,424 | 178 | 178 | **1,780** |
| **Positive** | 1,952 | 246 | 246 | **2,444** |

---

## Percentage Distribution per Class

| Class | Train % | Val % | Test % | Overall % |
|-------|---------|-------|--------|-----------|
| **Negative** | 63.9% | 63.7% | 63.8% | **63.8%** |
| **Neutral** | 15.2% | 15.3% | 15.2% | **15.2%** |
| **Positive** | 20.9% | 21.1% | 21.0% | **20.9%** |

---

## Dataset Sources

Three emotion datasets were used to train this model:

| Dataset | Train | Validation | Test | **Total** | Description |
|---------|-------|------------|------|-----------|-------------|
| **CREMA-D** | 5,959 | 749 | 734 | **7,442** | Crowd-sourced Emotional Multimodal Actors Dataset |
| **RAVDESS** | 1,161 | 141 | 138 | **1,440** | Ryerson Audio-Visual Database of Emotional Speech and Song |
| **TESS** | 2,225 | 277 | 298 | **2,800** | Toronto Emotional Speech Set |

---

## Key Insights

### 1. **Dataset Size**
- Total of **11,682 audio samples** across all splits
- Standard 80/10/10 train/validation/test split

### 2. **Class Imbalance**
- **Negative emotions dominate** at 63.8% of the dataset
- **Neutral emotions** are underrepresented at 15.2%
- **Positive emotions** make up 20.9%
- This imbalance is consistent across all splits

### 3. **Dataset Diversity**
- Uses **3 different emotion datasets** for better generalization
- **CREMA-D** is the largest contributor (63.7% of total data)
- **TESS** contributes 24.0% of the data
- **RAVDESS** contributes 12.3% of the data

### 4. **Model Performance Context**
From your evaluation results:
- **Overall Accuracy: 87.09%** on test set
- The model performs best on **negative** class (91.02% recall)
- **Neutral** class has good performance (85.96% recall)
- **Positive** class has lower recall (76.02%) - likely due to being underrepresented

---

## Recommendations

1. **Class Imbalance**: The model uses weighted sampling and class weights to handle the imbalance between negative/neutral/positive classes.

2. **Generalization**: Using 3 different datasets helps the model generalize better to different speakers and recording conditions.

3. **Test Set**: The test set (1,170 samples) is large enough to provide reliable performance estimates.

---

## Dataset Details

### CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- 7,442 samples total
- Multiple actors with diverse demographics
- Professional acted emotions

### RAVDESS (Ryerson Audio-Visual Database)
- 1,440 samples total
- High-quality studio recordings
- Includes both speech and song (likely only speech used here)

### TESS (Toronto Emotional Speech Set)
- 2,800 samples total
- Two female speakers
- 200 target words spoken in 7 emotions

---

*Report generated from manifest files in `data/processed/features/`*
