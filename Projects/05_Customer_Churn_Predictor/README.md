# Customer Churn Predictor

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Task](https://img.shields.io/badge/Task-Business%20Classification-2E8B57?style=flat-square)

## Project Summary

This project trains a binary classification model to predict whether a customer is likely to leave a service. It uses a synthetic customer dataset, which makes it useful for practicing business-style Machine Learning without needing private company data.

## Dataset

The dataset is generated with scikit-learn and includes practical customer behavior features.

| Feature | Meaning |
|---|---|
| `tenure_months` | How long the customer has stayed |
| `monthly_charges` | Current monthly billing amount |
| `support_calls` | Support interaction signal |
| `contract_score` | Contract strength signal |
| `usage_score` | Product usage signal |

Target:

```text
0 = customer stays
1 = customer churns
```

## Model Pipeline

```text
StandardScaler -> RandomForestClassifier
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
customer_churn_model.joblib
```

## What This Shows On GitHub

- Business classification problem
- Synthetic data generation
- Churn prediction workflow
- Precision, recall, and F1 score
- Saving and loading ML models

