# Wine Classifier

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Task](https://img.shields.io/badge/Task-Multiclass%20Classification-2E8B57?style=flat-square)

## Project Summary

This project trains a multiclass classifier to predict wine type from chemical measurements. It is a useful portfolio example because it works with many numeric features and multiple output classes.

## Dataset

The project uses the built-in Wine dataset from scikit-learn.

| Input Examples | Target |
|---|---|
| Alcohol, malic acid, ash, magnesium, flavanoids, color intensity, hue | Wine class |

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
wine_classifier_model.joblib
```

## What This Shows On GitHub

- Multiclass classification
- Random Forest modeling
- Feature-based prediction
- Classification metrics
- Model persistence

