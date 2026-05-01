# Breast Cancer Diagnosis Classifier

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic%20Regression-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-2E8B57?style=flat-square)

## Project Summary

This project trains a binary classification model to predict whether a tumor is malignant or benign using the Breast Cancer Wisconsin dataset from scikit-learn.

## Dataset

The dataset contains numeric measurements from cell nuclei, including radius, texture, perimeter, area, smoothness, and compactness.

| Type | Details |
|---|---|
| Dataset | Breast Cancer Wisconsin |
| Source | Built into scikit-learn |
| Target | Malignant or benign |

## Model Pipeline

```text
StandardScaler -> LogisticRegression
```

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

## What This Shows On GitHub

- Binary classification
- Healthcare-style evaluation
- Precision, recall, and F1 score
- Model pipelines
- Saving models with joblib

