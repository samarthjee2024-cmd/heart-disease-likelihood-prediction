"""
Disease Likelihood Prediction using Logistic Regression
Offline version: reads the bundled UCI Cleveland dataset from ./data/

Target mapping:
    Original target 0       -> No Disease (0)
    Original target 1,2,3,4 -> Disease (1)

Educational project only. This is not a medical diagnostic tool.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_FILE = Path(__file__).resolve().parent / "data" / "processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]


def load_heart_disease_data():
    """Load the Cleveland dataset bundled with this project; no internet needed."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_FILE}\n"
            "Keep the data folder and processed.cleveland.data file beside this script."
        )

    data = pd.read_csv(
        DATA_FILE,
        header=None,
        names=COLUMNS,
        na_values=["?", "NA", "na", ""],
        skipinitialspace=True,
    )
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Exclude rows with a missing target; impute missing predictor values in the pipeline.
    data = data.dropna(subset=["num"]).reset_index(drop=True)
    X = data.drop(columns=["num"])
    y = (data["num"] > 0).astype(int)

    if len(data) != 303 or X.shape[1] != 13:
        print(
            f"Note: loaded {len(data)} records and {X.shape[1]} features. "
            "The expected Cleveland subset has 303 records and 13 features."
        )
    if y.nunique() != 2:
        raise ValueError("Expected two classes after mapping target 0 vs. target 1–4.")
    return X, y


def main():
    try:
        X, y = load_heart_disease_data()
    except Exception as exc:
        print("Could not load the bundled dataset.")
        print("Please extract the entire ZIP and keep the data folder beside this script.")
        print(f"Details: {exc}")
        return

    print("Dataset loaded locally — no internet connection required.")
    print(f"Dataset: {len(X)} records, {X.shape[1]} features")
    print(f"No Disease: {(y == 0).sum()} | Disease: {(y == 1).sum()}")
    print("Features:", ", ".join(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Fit imputation and scaling on training data only to avoid test-data leakage.
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    print("\n========== MODEL EVALUATION ==========")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")
    print(f"Accuracy        : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"ROC-AUC         : {roc_auc_score(y_test, y_probability):.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["No Disease", "Disease"], zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion Matrix (rows = actual, columns = predicted):")
    print("                 Predicted No Disease | Predicted Disease")
    print(f"Actual No Disease       {cm[0, 0]:>5}              {cm[0, 1]:>5}")
    print(f"Actual Disease          {cm[1, 0]:>5}              {cm[1, 1]:>5}")

    print("\n========== TRY A SAMPLE PREDICTION ==========")
    print("Enter 'n' to skip, or 'y' to enter 13 numeric feature values.")
    choice = input("Would you like to enter a patient record? (y/n): ").strip().lower()
    if choice == "y":
        print("\nFeature order:")
        for index, name in enumerate(X.columns, start=1):
            print(f"{index}. {name}")
        print("\nEnter 13 numeric values separated by commas, using the dataset's category codes.")
        raw = input("Values: ").strip()
        try:
            values = [float(value.strip()) for value in raw.split(",")]
            if len(values) != X.shape[1]:
                raise ValueError(f"You must enter exactly {X.shape[1]} values.")
            sample = pd.DataFrame([values], columns=X.columns)
            probability = float(model.predict_proba(sample)[0, 1])
            label = "Disease" if probability >= 0.5 else "No Disease"
            print(f"\nPredicted class: {label}")
            print(f"Estimated model probability for Disease: {probability * 100:.2f}%")
            print("Educational model output only—not a diagnosis or personal risk estimate.")
        except ValueError as exc:
            print(f"Could not make a prediction: {exc}")
            print("Rerun the script and enter exactly 13 comma-separated numeric values.")
    else:
        print("Sample prediction skipped.")
    print("\nDone.")


if __name__ == "__main__":
    main()
