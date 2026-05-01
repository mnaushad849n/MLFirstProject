# Iris Flower Classifier

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic%20Regression-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Task](https://img.shields.io/badge/Task-Multiclass%20Classification-2E8B57?style=flat-square)

## Project Summary

This project trains a Machine Learning model to classify Iris flowers into three species: setosa, versicolor, and virginica. It is a clean beginner project for learning the full classification workflow.

## Dataset

The project uses the built-in Iris dataset from scikit-learn.

| Input Features | Target |
|---|---|
| Sepal length, sepal width, petal length, petal width | Iris species |

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
iris_model.joblib
```

## What This Shows On GitHub

- Multiclass classification
- Train/test split
- Feature scaling
- Logistic Regression
- Model evaluation
- Saving and loading models

