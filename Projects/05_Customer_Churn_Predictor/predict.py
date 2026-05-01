import joblib
import pandas as pd


MODEL_FILE = "customer_churn_model.joblib"


def main():
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    target_names = saved["target_names"]
    feature_names = saved["feature_names"]

    sample = pd.DataFrame([[0.4, 1.2, 1.8, -0.6, -1.1]], columns=feature_names)
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    print(sample)
    print("Predicted customer status:", target_names[prediction])
    print("Confidence:", round(probability[prediction], 4))


if __name__ == "__main__":
    main()
