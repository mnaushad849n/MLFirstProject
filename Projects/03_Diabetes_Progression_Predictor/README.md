# Diabetes Progression Predictor

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Ridge%20Regression-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Task](https://img.shields.io/badge/Task-Regression-2E8B57?style=flat-square)

## Project Summary

This project trains a regression model to predict a numeric diabetes disease progression score from patient measurements.

## Dataset

The project uses the built-in Diabetes dataset from scikit-learn.

| Type | Details |
|---|---|
| Dataset | Diabetes |
| Source | Built into scikit-learn |
| Target | Numeric disease progression score |

## Model Pipeline

```text
StandardScaler -> Ridge Regression
```

## How To Run

```bash
pip install -r requirements.txt
python train_model.py
python predict.py
```

## Output

Training prints:

- MAE
- RMSE
- R2 score

It saves:

```text
diabetes_regression_model.joblib
```

## What This Shows On GitHub

- Regression workflow
- Numeric prediction
- MAE, RMSE, and R2 evaluation
- Ridge Regression
- Saving regression models

