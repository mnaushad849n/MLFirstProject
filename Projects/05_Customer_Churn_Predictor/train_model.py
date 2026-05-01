import joblib
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_FILE = "customer_churn_model.joblib"
RANDOM_STATE = 42
FEATURE_NAMES = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "contract_score",
    "usage_score",
]


def create_dataset():
    X, y = make_classification(
        n_samples=1200,
        n_features=len(FEATURE_NAMES),
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        weights=[0.7, 0.3],
        class_sep=1.2,
        random_state=RANDOM_STATE,
    )
    X = pd.DataFrame(X, columns=FEATURE_NAMES)
    return X, y


def main():
    X, y = create_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(classification_report(y_test, predictions, target_names=["stay", "churn"]))

    joblib.dump({
        "model": model,
        "target_names": ["stay", "churn"],
        "feature_names": FEATURE_NAMES,
    }, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
