# 🎙️ Deepfake Audio Detection
### MARS Open Projects 2026 — Problem Statement 2

![Accuracy](https://img.shields.io/badge/Accuracy-≥80%25-brightgreen)
![EER](https://img.shields.io/badge/EER-≤12%25-brightgreen)
![F1](https://img.shields.io/badge/F1-≥80%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)

> A machine learning system that classifies speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)**.

---

## 📌 Table of Contents
1. [Project Description](#-project-description)
2. [Architecture](#-architecture)
3. [Methodology](#-methodology)
4. [Metric Results](#-metric-results)
5. [Repository Structure](#-repository-structure)
6. [Setup Instructions](#-setup-instructions)
7. [How to Run](#-how-to-run)
8. [Streamlit Web App](#-streamlit-web-app)

---

## 📖 Project Description

Advances in generative AI have enabled creation of highly realistic synthetic speech. Such audio can be misused for impersonation, fraud, and social engineering.

This system detects whether a speech recording is **Genuine** or **AI-Generated (Deepfake)** using a rich feature extraction pipeline combined with an ensemble classifier.

**Two types of detection:**
- ✅ **Genuine (Human)** — Natural human speech patterns
- 🚨 **Deepfake (AI-Generated)** — Synthetic artifacts inconsistent with real speech

---

## 🏗 Architecture

```
Raw Audio (.wav / .flac)
         ↓
   Stage 0: EDA & Manifest
         ↓
   Stage 1: Feature Extraction
   ┌─────────────────────────────────────┐
   │  MFCC (40) + Delta + Delta-Delta    │
   │  Mel Spectrogram (64 bins)          │
   │  Chroma (12 bins)                   │
   │  Spectral Centroid / BW / Rolloff   │
   │  Spectral Contrast / Flatness       │
   │  Zero Crossing Rate                 │
   │  RMS Energy + Tonnetz               │
   │  Total: ~280 features per file      │
   └─────────────────────────────────────┘
         ↓
   Stage 2: Ensemble Classifier
   ┌─────────────────────────────────────┐
   │  Random Forest (100 trees)          │
   │  + Gradient Boosting (200 est.)     │
   │  + Logistic Regression (calibrated) │
   │  → Soft-vote ensemble               │
   └─────────────────────────────────────┘
         ↓
   Prediction + Confidence Score
         ↓
   Streamlit Web App
```

---

## 🔬 Methodology

### Stage 0 — EDA & Data Manifest
- Walks the dataset directory and builds a manifest CSV
- Reports class distribution, duration stats, sample rate distribution
- Identifies short/long files that may affect quality

### Stage 1 — Feature Extraction
Audio is resampled to **16kHz**, padded/clipped to **4 seconds**. Features extracted per file:

| Feature Group | Details | Dimensions |
|---|---|---|
| MFCC | 40 coefficients × mean + std | 80 |
| MFCC Delta | First-order temporal difference | 80 |
| MFCC Delta-Delta | Second-order temporal difference | 80 |
| Mel Spectrogram | 64 bins × mean/std/max/min | 4 |
| Chroma | 12 pitch class profiles × mean + std | 24 |
| Spectral Centroid | Mean + std | 2 |
| Spectral Bandwidth | Mean + std | 2 |
| Spectral Rolloff | Mean + std | 2 |
| Spectral Contrast | Mean + std | 2 |
| Spectral Flatness | Mean + std | 2 |
| Zero Crossing Rate | Mean + std | 2 |
| RMS Energy | Mean + std | 2 |
| Tonnetz | Mean + std | 2 |
| **Total** | | **~286** |

### Stage 2 — Model Training
- **Class Imbalance:** SMOTE oversampling + weighted loss
- **Ensemble:** Random Forest + Gradient Boosting + Logistic Regression (soft-vote)
- **Threshold Optimization:** Best decision threshold found by maximizing Macro F1
- **Evaluation:** Accuracy, EER, F1, Per-Class Accuracy, Confusion Matrix

### EER Calculation
Equal Error Rate = point where False Acceptance Rate = False Rejection Rate.
Computed via ROC curve interpolation using `scipy.optimize.brentq`.

---

## 📈 Metric Results

| Metric | Required | Achieved |
|--------|----------|----------|
| Overall Accuracy | ≥ 80% | *see deepfake_metrics.json* |
| Equal Error Rate | ≤ 12% | *see deepfake_metrics.json* |
| F1 Score (macro) | ≥ 80% | *see deepfake_metrics.json* |
| Per-Class Accuracy (Real) | ≥ 75% | *see deepfake_metrics.json* |
| Per-Class Accuracy (Fake) | ≥ 75% | *see deepfake_metrics.json* |

> Run `python train_pipeline.py` to populate these values.

Full performance report: `deepfake_report.txt`

---

## 📁 Repository Structure

```
deepfake-audio-detection/
│
├── deepfake_eda.py          # Stage 0: EDA & data manifest
├── deepfake_features.py     # Stage 1: Feature extraction
├── deepfake_train.py        # Stage 2: Training + evaluation
├── deepfake_predict.py      # Inference script for new audio files
│
├── train_pipeline.py        # Full pipeline runner (all stages)
├── app.py                   # Streamlit web app
├── notebook.ipynb           # Full reproducible notebook
│
├── requirements.txt
├── README.md
│
└── data/                    # Dataset folder (download from Kaggle)
    ├── train/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
```

---

## ⚙️ Setup Instructions

### 1. Clone repository
```bash
git clone https://github.com/<your-username>/deepfake-audio-detection.git
cd deepfake-audio-detection
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
Go to: https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

Extract and organize as:
```
data/
├── train/
│   ├── real/   ← genuine .wav files
│   └── fake/   ← deepfake .wav files
└── test/
    ├── real/
    └── fake/
```

---

## 🚀 How to Run

### Full Pipeline (Recommended)
```bash
python train_pipeline.py
```

### Run Stages Individually
```bash
python deepfake_eda.py          # Stage 0: EDA
python deepfake_features.py     # Stage 1: Extract features
python deepfake_train.py        # Stage 2: Train + evaluate
```

### Inference on New Audio
```bash
# Single file
python deepfake_predict.py --input audio.wav

# Folder of files
python deepfake_predict.py --input audio_folder/ --output results.csv
```

### Skip Feature Extraction (Re-use existing)
```bash
python train_pipeline.py --skip_features
```

---

## 🌐 Streamlit Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501`

**Features:**
- Upload audio file (.wav / .flac / .mp3)
- Instant prediction: Genuine or Deepfake
- Confidence score with visual bar
- Individual classifier breakdown (RF / GB / LR scores)
- Feature analysis summary
- Model performance metrics display

**Hosted demo:** `https://<your-deployment>.streamlit.app`

---

## 📦 Key Libraries

| Library | Purpose |
|---------|---------|
| `librosa` | Audio loading + feature extraction |
| `scikit-learn` | Random Forest, GBM, Logistic Regression |
| `imbalanced-learn` | SMOTE for class imbalance |
| `scipy` | EER computation via ROC interpolation |
| `streamlit` | Web application |
| `soundfile` | Audio file I/O |
