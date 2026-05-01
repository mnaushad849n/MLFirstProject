# Customer Churn Predictor

This project trains a binary classification model to predict whether a customer is likely to leave a service.

## Dataset

This project creates a synthetic customer dataset using scikit-learn. It is useful for practice when you do not have a real business dataset yet.

Example features:

- Tenure
- Monthly charges
- Support calls
- Contract score
- Usage score

Target:

- `0`: customer stays
- `1`: customer churns

## Model

The model uses:

1. `StandardScaler`
2. `RandomForestClassifier`

## How To Run

```bash
pip install -r requirements.txt
python train_model.py
python predict.py
```

## Output

Training prints accuracy and classification report, then saves:

```text
customer_churn_model.joblib
```

## What You Learn

- Business classification problem
- Synthetic data generation
- Churn prediction workflow
- Precision, recall, and F1 score
- Saving and loading ML models

