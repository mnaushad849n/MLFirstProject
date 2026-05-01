import joblib
import pandas as pd


MODEL_FILE = "breast_cancer_model.joblib"


def main():
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    target_names = saved["target_names"]
    feature_names = saved["feature_names"]

    sample_values = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
        0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
        184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
    ]
    sample = pd.DataFrame([sample_values], columns=feature_names)

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    print("Predicted diagnosis:", target_names[prediction])
    print("Confidence:", round(probability[prediction], 4))


if __name__ == "__main__":
    main()
