import joblib
import pandas as pd


MODEL_FILE = "diabetes_regression_model.joblib"


def main():
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    feature_names = saved["feature_names"]

    sample = pd.DataFrame(
        [[0.038, 0.051, 0.062, 0.022, -0.044, -0.034, -0.043, -0.003, 0.019, -0.017]],
        columns=feature_names,
    )

    prediction = model.predict(sample)[0]
    print("Predicted disease progression score:", round(prediction, 2))


if __name__ == "__main__":
    main()
