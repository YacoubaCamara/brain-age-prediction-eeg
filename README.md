# Brain Age Prediction from EEG

A machine learning pipeline that predicts brain age from resting-state EEG signals using frequency band power features.

## Overview

This project extracts EEG band power features (delta, theta, alpha, beta, gamma) from raw EEG recordings and trains multiple machine learning models to predict a subject's age from their brain activity patterns.

## Dataset

[EEG for Age Prediction](https://kaggle.com/datasets/ayurgo/data-eeg-age-v1) — 126 subjects, ages 18–89.

Download the eval folder and place the CSV files in a `data/` directory.

## Pipeline

1. Load EEG CSV files and extract age labels
2. Compute absolute and relative band power features using Welch's method
3. Train and evaluate 5 models with Leave-One-Out cross-validation
4. Visualize results

## Results

<img width="1200" height="600" alt="model_comparison" src="https://github.com/user-attachments/assets/c72f8454-1f5b-4b0f-af26-a299daf0b1a9" />
<img width="1800" height="750" alt="brain_age_gap" src="https://github.com/user-attachments/assets/a887ea3f-b0a8-48e9-84b7-21aca73a9987" />
<img width="1200" height="750" alt="feature_importance" src="https://github.com/user-attachments/assets/19b2faad-e588-4a75-ae38-433a409f5595" />

Best model: **Random Forest** with MAE of 15.11 years.

## Setup

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python demo.py
```

## Requirements

See `requirements.txt`
