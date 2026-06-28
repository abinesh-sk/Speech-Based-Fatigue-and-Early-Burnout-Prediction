# Running Model Validation

This guide explains how to evaluate your trained CRNN model with detailed metrics and confusion matrix visualization.

## Quick Start

### 1. Install Dependencies

First, activate your virtual environment and install all required packages:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

### 2. Run Validation

To evaluate your trained model (`crnn_best.pt`) on the **test set**:

```powershell
python -m src.validate_detailed --checkpoint crnn_best.pt --split test
```

## Command Options

```powershell
python -m src.validate_detailed [OPTIONS]
```

### Available Options:

- `--checkpoint` : Path to model checkpoint (default: `crnn_best.pt`)
- `--split` : Dataset split to evaluate - choices: `train`, `val`, `test` (default: `test`)
- `--batch_size` : Batch size for evaluation (default: `16`)
- `--num_workers` : Number of data loading workers (default: `0`)
- `--hidden_size` : GRU hidden size (default: `192`)
- `--dropout` : Dropout rate (default: `0.3`)
- `--output_dir` : Directory to save confusion matrix image (default: `.`)

### Examples:

**Evaluate on test set (recommended):**
```powershell
python -m src.validate_detailed --checkpoint crnn_best.pt --split test
```

**Evaluate on validation set:**
```powershell
python -m src.validate_detailed --checkpoint crnn_best.pt --split val
```

**Save confusion matrix to specific directory:**
```powershell
python -m src.validate_detailed --checkpoint crnn_best.pt --split test --output_dir ./results
```

## What You'll Get

### 1. Terminal Output

The script will display:

- **Overall Accuracy**: Total accuracy percentage
- **Cohen's Kappa**: Agreement metric (0-1 scale)
- **Per-Class Metrics**: Precision, Recall, F1-Score, and Support for each emotion class
  - `negative`
  - `neutral`
  - `positive`
- **Macro & Weighted Averages**: Aggregate metrics across all classes
- **Detailed Classification Report**: Comprehensive sklearn classification report
- **Confusion Matrix (Text)**: ASCII table showing prediction counts

### 2. Confusion Matrix Image

A high-resolution PNG image (`confusion_matrix_test.png`) will be saved showing:

- **Left panel**: Confusion matrix with raw counts
- **Right panel**: Confusion matrix with percentages (normalized by true labels)

The image is saved at 300 DPI for publication quality.

## Understanding the Metrics

### Accuracy
- Overall percentage of correct predictions
- Formula: (Correct Predictions) / (Total Predictions)

### Precision
- Of all samples predicted as class X, how many were actually class X?
- Formula: True Positives / (True Positives + False Positives)

### Recall (Sensitivity)
- Of all actual class X samples, how many were correctly predicted?
- Formula: True Positives / (True Positives + False Negatives)

### F1-Score
- Harmonic mean of Precision and Recall
- Formula: 2 × (Precision × Recall) / (Precision + Recall)

### Cohen's Kappa
- Measures agreement between predictions and ground truth, accounting for chance
- Range: -1 to 1 (higher is better, >0.8 is excellent)

### Support
- Number of actual samples in each class

## Troubleshooting

### Error: "No module named 'src'"

Make sure you're running the command from the project root directory:
```powershell
cd c:\Users\kirut\Desktop\VoiceAnalysis\VoiceAnalysis
```

### Error: "FileNotFoundError: manifest.json"

Ensure your data has been preprocessed and feature extraction has been completed. The following files should exist:
- `data/processed/features/train/manifest.json`
- `data/processed/features/val/manifest.json`
- `data/processed/features/test/manifest.json`

### Error: "No such file: crnn_best.pt"

Make sure the model checkpoint exists in the current directory, or specify the correct path:
```powershell
python -m src.validate_detailed --checkpoint path/to/your/model.pt
```

## Model Architecture

Your model uses:
- **CNN**: 3 convolutional blocks for feature extraction
- **GRU**: Bidirectional GRU for temporal modeling
- **Classes**: 3 emotion polarities (negative, neutral, positive)
- **Hidden Size**: 192 (default)
