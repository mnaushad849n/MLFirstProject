import joblib
import pandas as pd


MODEL_FILE = "wine_classifier_model.joblib"


def main():
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    target_names = saved["target_names"]
    feature_names = saved["feature_names"]

    sample_values = [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0]
    sample = pd.DataFrame([sample_values], columns=feature_names)

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    print("Predicted wine class:", target_names[prediction])
    print("Confidence:", round(probability[prediction], 4))


if __name__ == "__main__":
    main()
