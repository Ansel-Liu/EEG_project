# Spatiotemporal Deep Learning for Epileptic Seizure Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-Data%20Engineering-150458.svg)](https://pandas.pydata.org/)

## Project Overview

This repository contains an end-to-end, modular Machine Learning pipeline designed to detect epileptic seizures from multi-channel EEG signals.

Because continuous EEG monitoring presents a highly imbalanced classification problem with significant patient-level data shifts, this project implements a **Hybrid 1D CNN + Bidirectional LSTM** architecture. It effectively extracts spatial morphological features across channels and captures long-term temporal dependencies to accurately identify seizure onset.

---

## Core Engineering Highlights

### 1. Data Engineering & Leakage Prevention

- **Defensive Data I/O:** Decoupled raw signal loading (`.npz`) and metadata parsing (`.parquet`) using **Abstract Base Classes (ABC)** following SOLID principles, ensuring scalability and format independence.

- **Strict Patient-Level Splits:** Utilized `GroupKFold` to guarantee that EEG windows from the same patient never appear in both training and testing sets, completely eliminating temporal data leakage.

- **Signal Preprocessing:** Implemented configurable 1D Median Smoothing filters to remove high-frequency muscle artifacts and applied independent Z-score normalization across the `(Samples × Channels)` dimension.

### 2. Spatiotemporal Modeling

- **Late Feature Fusion:** Designed a custom PyTorch architecture that independently processes 21 EEG channels through a 1D-CNN backbone before concatenating the extracted spatial features.

- **Temporal Dynamics:** Integrated a `Bidirectional LSTM` layer with sequence-level average pooling to model forward and backward temporal dependencies without overfitting to the final time step.

### 3. Automated Training & Evaluation

- **Imbalance Handling:** Dynamically generated positive class weights (`pos_weight`) together with `BCEWithLogitsLoss` to improve numerical stability. Implemented metric-driven early stopping based on validation F1-score.

- **Automated Reporting:** Aggregated out-of-fold predictions, generated PR-AUC and ROC-AUC curves, saved confusion matrices (TN, FP, FN, TP), and exported LaTeX snippets for academic reporting.

---

## Repository Structure

```text
├── config/                 # Hyperparameter configurations (DataClasses)
├── data_profiling/         # Automated EDA and dataset statistical reports
├── data_utils/             # Preprocessing, Z-score scaling, and dataset splits
├── io_data/                # Defensive I/O loaders for NPZ and Parquet files
├── models/                 # Neural network architectures (Feature Fusion CNN, Bi-LSTM)
├── training/               # Training loops, early stopping, and evaluation metrics
├── figures/                # Auto-generated visualization charts
├── results/                # Aggregated confusion matrices and performance CSVs
└── run_baseline.py         # Main execution entry point
```

---

## Key Results

Our Feature Fusion CNN-LSTM model effectively handles spatial-temporal variations in unseen patients during 5-fold cross-validation.

| Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---------------|----------:|----------:|--------:|---------:|--------:|-------:|
| Feature Fusion CNN-LSTM | 0.8720 | 0.3277 | 0.3993 | 0.3600 | 0.8631 | 0.4142 |

> **Note:** The F1-score is heavily affected by the extreme class imbalance inherent in continuous EEG monitoring. Threshold tuning prioritizes high ROC-AUC and robust recall, which are critical for real-world clinical alarm systems.

---

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- SciPy
- Scikit-Learn
- GroupKFold Cross Validation
- BCEWithLogitsLoss
- Bidirectional LSTM
- 1D CNN
- Time-Series Analysis

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/YourRepository.git
cd YourRepository
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the dataset path

Update the `data_dir` variable inside `config/config.py` (or pass it through command-line arguments) to point to your local `.npz` and `.parquet` files.

### 4. Run the complete pipeline

```bash
python run_baseline.py
```

---



