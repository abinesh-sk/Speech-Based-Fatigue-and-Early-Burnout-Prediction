# Voice Analysis Model Evaluation - Complete Setup

## ✅ Installation Complete!

All necessary packages have been installed successfully:
- **PyTorch** (Deep Learning framework)
- **Librosa** (Audio processing)
- **Scikit-learn** (Metrics and evaluation)
- **Matplotlib & Seaborn** (Visualization)
- **NumPy, SoundFile, Resampy** (Data processing)
- **tqdm** (Progress bars)

---

## 🚀 How to Run Model Evaluation

### Step 1: Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2: Run the Detailed Validation Script

To evaluate your trained model on the **test set** with full metrics and confusion matrix:

```powershell
python -m src.validate_detailed --checkpoint crnn_best.pt --split test
```

**Alternative splits:**
- `--split train` - Evaluate on training set
- `--split val` - Evaluate on validation set
- `--split test` - Evaluate on test set (recommended)

### Step 3: View Results

The script will:
1. **Print to terminal:**
   - Overall accuracy
   - Cohen's Kappa score
   - Per-class metrics (Precision, Recall, F1-Score)
   - Confusion matrix (text format)
   - Detailed classification report

2. **Save confusion matrix image:**
   - File: `confusion_matrix_test.png` (or `_val`, `_train` depending on split)
   - Location: Current directory (or specify with `--output_dir`)
   - Format: High-resolution PNG (300 DPI)
   - Contains: Both count and percentage views

---

## 📊 What Metrics You'll See

### Overall Metrics
- **Accuracy**: Percentage of correct predictions
- **Cohen's Kappa**: Agreement metric (0-1, higher is better)

### Per-Class Metrics (for each emotion: negative, neutral, positive)
- **Precision**: How many predicted samples were correct
- **Recall**: How many actual samples were found
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples in each class

### Confusion Matrix
- **Rows**: True labels (actual emotions)
- **Columns**: Predicted labels
- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications

---

## 📝 Example Output

```
============================================================
EVALUATION METRICS
============================================================

Overall Accuracy:              0.8542 (85.42%)
Cohen's Kappa:                 0.7813

------------------------------------------------------------
PER-CLASS METRICS
------------------------------------------------------------
Class           Precision    Recall       F1-Score     Support   
------------------------------------------------------------
negative        0.8234       0.8567       0.8397       234       
neutral         0.8912       0.8456       0.8678       345       
positive        0.8456       0.8923       0.8683       267       
------------------------------------------------------------
Macro Avg       0.8534       0.8649       0.8586       846       
Weighted Avg    0.8556       0.8542       0.8545       846       
------------------------------------------------------------

Confusion matrix saved to:     ./confusion_matrix_test.png
```

---

## 🔧 Advanced Options

### Custom Output Directory
```powershell
python -m src.validate_detailed --checkpoint crnn_best.pt --split test --output_dir ./results
```

### Different Model Configuration
If your model was trained with different hyperparameters:
```powershell
python -m src.validate_detailed --checkpoint crnn_best.pt --split test --hidden_size 256 --dropout 0.5
```

### Larger Batch Size (faster evaluation)
```powershell
python -m src.validate_detailed --checkpoint crnn_best.pt --split test --batch_size 32
```

---

## 🎯 Model Information

**Architecture:** CRNN (Convolutional Recurrent Neural Network)
- **CNN Layers**: 3 convolutional blocks with BatchNorm and MaxPooling
- **RNN Layer**: Bidirectional GRU (Gated Recurrent Unit)
- **Hidden Size**: 192 (default)
- **Output Classes**: 3 (negative, neutral, positive)

**Input:** Mel-spectrogram features (80 mel bands)
**Task:** Emotion polarity classification from speech

---

## ❓ Troubleshooting

### "No module named 'src'"
Make sure you're in the project root directory:
```powershell
cd c:\Users\kirut\Desktop\VoiceAnalysis\VoiceAnalysis
```

### "FileNotFoundError: manifest.json"
Your data needs to be preprocessed first. The following files must exist:
- `data/processed/features/test/manifest.json`
- `data/processed/features/val/manifest.json`
- `data/processed/features/train/manifest.json`

### "No such file: crnn_best.pt"
Specify the correct path to your model checkpoint:
```powershell
python -m src.validate_detailed --checkpoint path/to/your/model.pt
```

### Package Import Errors
Verify installation:
```powershell
python check_installation.py
```

If packages are missing:
```powershell
pip install -r requirements.txt
```

---

## 📁 Files Created

1. **`src/validate_detailed.py`** - Enhanced validation script
2. **`requirements.txt`** - Updated with all dependencies
3. **`VALIDATION_GUIDE.md`** - Detailed validation guide
4. **`check_installation.py`** - Installation verification script
5. **`SETUP_COMPLETE.md`** - This file

---

## 🎉 You're Ready!

Everything is set up. Just run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.validate_detailed --checkpoint crnn_best.pt --split test
```

The confusion matrix image will be saved in the current directory, and all metrics will be displayed in the terminal.

Good luck with your evaluation! 🚀
