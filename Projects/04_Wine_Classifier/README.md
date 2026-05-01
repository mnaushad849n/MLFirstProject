# Wine Classifier

This project trains a multiclass classifier to predict wine type from chemical measurements.

## Dataset

The project uses the built-in Wine dataset from scikit-learn.

Features include alcohol, malic acid, ash, magnesium, flavanoids, color intensity, hue, and other chemical measurements.

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
wine_classifier_model.joblib
```

## What You Learn

- Multiclass classification
- Random Forest model
- Feature-based prediction
- Classification metrics
- Model persistence

