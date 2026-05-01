# Diabetes Progression Predictor

This project trains a regression model to predict diabetes disease progression from patient measurements.

## Dataset

The project uses the built-in Diabetes dataset from scikit-learn.

The target is a numeric disease progression score.

## Model

The model uses:

1. `StandardScaler`
2. `Ridge` regression

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

## What You Learn

- Regression
- Numeric prediction
- MAE, RMSE, and R2
- Ridge Regression
- Saving regression models

