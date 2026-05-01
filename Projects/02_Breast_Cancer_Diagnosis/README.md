# Breast Cancer Diagnosis Classifier

This project trains a binary classification model to predict whether a breast cancer tumor is malignant or benign.

## Dataset

The project uses the built-in Breast Cancer Wisconsin dataset from scikit-learn.

The dataset contains numeric measurements from cell nuclei, such as radius, texture, perimeter, area, smoothness, and compactness.

## Model

The model uses:

1. `StandardScaler`
2. `LogisticRegression`

## Why This Project Matters

This is a common beginner healthcare-style ML problem. It teaches why precision, recall, and F1 score can matter more than only accuracy.

## How To Run

```bash
pip install -r requirements.txt
python train_model.py
python predict.py
```

## Output

Training prints accuracy and a classification report, then saves:

```text
breast_cancer_model.joblib
```

## What You Learn

- Binary classification
- Healthcare-style evaluation
- Precision and recall
- Model pipelines
- Saving models with joblib

