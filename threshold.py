# test_model_performance.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def test_model_accuracy_threshold():
    model = joblib.load("machine_failure_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    # Load a held-out test set (you'd save this separately during training)
    df = pd.read_csv("processed_machine_data.csv")
    X_test = df[feature_columns]
    y_test = df["Machine failure"]

    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # This is the "threshold" — the minimum acceptable performance
    assert accuracy >= 0.80, f"Accuracy dropped to {accuracy:.2f}, below 80% threshold"
    assert f1 >= 0.60, f"F1 score dropped to {f1:.2f}, below 60% threshold"