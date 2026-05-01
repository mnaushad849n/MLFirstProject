# Iris Flower Classifier

This project trains a Machine Learning model to classify Iris flowers into three species: setosa, versicolor, and virginica.

## Dataset

The project uses the built-in Iris dataset from scikit-learn.

Features:

- Sepal length
- Sepal width
- Petal length
- Petal width

Target:

- Iris species

## Model

The model uses a scikit-learn pipeline:

1. `StandardScaler`
2. `LogisticRegression`

## How To Run

```bash
pip install -r requirements.txt
python train_model.py
python predict.py
```

## Output

Training prints classification metrics and saves:

```text
iris_model.joblib
```

## What You Learn

- Multiclass classification
- Train/test split
- Feature scaling
- Logistic Regression
- Saving and loading models

