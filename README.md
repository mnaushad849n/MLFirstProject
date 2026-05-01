<div align="center">

# ML First Project

### Beginner-friendly Machine Learning projects with clean code, saved models, and reproducible workflows

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Portfolio%20Ready-2E8B57?style=for-the-badge)

</div>

## Overview

This repository contains a small but complete collection of beginner Machine Learning projects. Each project follows a practical workflow: load data, split the dataset, train a model, evaluate performance, save the trained model, and run a separate prediction script.

It is designed to show consistent project structure on GitHub and to make each ML example easy to run, explain, and extend.

## Projects Included

| No. | Project | Problem Type | Dataset | Main Skill |
|---:|---|---|---|---|
| 1 | [Iris Flower Classifier](Projects/01_Iris_Flower_Classifier) | Classification | scikit-learn Iris | Logistic Regression pipeline |
| 2 | [Breast Cancer Diagnosis](Projects/02_Breast_Cancer_Diagnosis) | Classification | Breast Cancer Wisconsin | Binary classification metrics |
| 3 | [Diabetes Progression Predictor](Projects/03_Diabetes_Progression_Predictor) | Regression | scikit-learn Diabetes | Numeric prediction with Ridge |
| 4 | [Wine Classifier](Projects/04_Wine_Classifier) | Classification | scikit-learn Wine | Random Forest multiclass modeling |
| 5 | [Customer Churn Predictor](Projects/05_Customer_Churn_Predictor) | Classification | Synthetic business data | Churn workflow and evaluation |

## Repository Structure

```text
GitHub_MLFirstProject/
|-- README.md
|-- requirements.txt
|-- train_model.py
|-- predict.py
|-- iris_model.joblib
`-- Projects/
    |-- 01_Iris_Flower_Classifier/
    |-- 02_Breast_Cancer_Diagnosis/
    |-- 03_Diabetes_Progression_Predictor/
    |-- 04_Wine_Classifier/
    `-- 05_Customer_Churn_Predictor/
```

## Quick Start

Open any project folder inside `Projects/`, install the requirements, train the model, and run prediction.

```bash
cd Projects/01_Iris_Flower_Classifier
pip install -r requirements.txt
python train_model.py
python predict.py
```

## Common Workflow

| Step | What Happens |
|---:|---|
| 1 | Load or create the dataset |
| 2 | Split data into train and test sets |
| 3 | Build a scikit-learn pipeline |
| 4 | Train the model |
| 5 | Print evaluation metrics |
| 6 | Save the model with joblib |
| 7 | Load the model in `predict.py` |

## Skills Demonstrated

- Loading datasets with scikit-learn
- Building reusable ML pipelines
- Feature scaling with `StandardScaler`
- Classification and regression modeling
- Evaluating accuracy, precision, recall, F1, MAE, RMSE, and R2
- Saving and loading trained models with `joblib`
- Writing simple prediction scripts for portfolio projects

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| pandas | Data handling |
| scikit-learn | Datasets, models, pipelines, metrics |
| joblib | Model persistence |

## Next Improvements

- Add Jupyter notebook versions
- Add charts and confusion matrices
- Add Streamlit web apps
- Add model comparison tables
- Add screenshots and sample outputs

