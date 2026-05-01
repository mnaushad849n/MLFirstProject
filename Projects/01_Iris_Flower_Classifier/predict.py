import joblib
import pandas as pd


MODEL_FILE = "iris_model.joblib"


def main():
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    target_names = saved["target_names"]
    feature_names = saved["feature_names"]

    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=feature_names)
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    print(sample)
    print("Predicted species:", target_names[prediction])
    print("Confidence:", round(probability[prediction], 4))


if __name__ == "__main__":
    main()
