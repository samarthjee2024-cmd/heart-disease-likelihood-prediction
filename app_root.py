"""
Heart Disease Likelihood Prediction — educational Streamlit demo.
Uses the bundled UCI Cleveland Heart Disease dataset; no runtime dataset download.
Not a medical device, diagnostic tool, or personal clinical risk assessment.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Heart Disease Likelihood Demo",
    page_icon="🫀",
    layout="wide",
)

DATA_FILE = Path(__file__).resolve().parent / "processed.cleveland.data"
RANDOM_STATE = 42
TEST_SIZE = 0.20
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]
FEATURES = COLUMNS[:-1]


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "The bundled dataset is missing. Keep processed.cleveland.data in the repository root "
            "inside the project folder."
        )
    df = pd.read_csv(
        DATA_FILE, header=None, names=COLUMNS,
        na_values=["?", "NA", "na", ""], skipinitialspace=True
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["num"]).reset_index(drop=True)
    X = df[FEATURES]
    y = (df["num"] > 0).astype(int)
    return X, y


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


@st.cache_resource
def train_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    evaluation_model = make_model()
    evaluation_model.fit(X_train, y_train)
    test_pred = evaluation_model.predict(X_test)
    test_prob = evaluation_model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    auc = roc_auc_score(y_test, test_prob)

    # Fit a separate model on all available records for the interactive demo.
    # Holdout metrics above still come only from the untouched test split.
    full_model = make_model()
    full_model.fit(X, y)
    metrics = {
        "n": len(X),
        "n_features": X.shape[1],
        "train_n": len(X_train),
        "test_n": len(X_test),
        "accuracy": accuracy_score(y_test, test_pred),
        "auc": auc,
        "report": classification_report(
            y_test, test_pred, labels=[0, 1],
            target_names=["No Disease", "Disease"],
            output_dict=True, zero_division=0,
        ),
        "cm": cm,
        "class_counts": y.value_counts().sort_index().to_dict(),
    }
    return full_model, metrics


st.title("🫀 Heart Disease Likelihood Prediction")
st.caption("Logistic Regression · UCI Cleveland dataset · Educational machine-learning demonstration")

st.warning(
    "**Important:** This is an academic demonstration trained on a small, historical dataset. "
    "It is not a diagnosis, does not estimate your personal clinical risk, and must not be used "
    "to make healthcare decisions. If you have health concerns, consult a qualified healthcare professional."
)

try:
    X, y = load_data()
    prediction_model, metrics = train_models()
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.stop()

with st.expander("About the dataset and model", expanded=False):
    st.write(
        "The app uses the Cleveland subset of the UCI Heart Disease dataset. The original target "
        "values are grouped as 0 = No Disease and 1–4 = Disease. Missing predictor values are "
        "median-imputed inside a scikit-learn pipeline, followed by standardization and Logistic Regression."
    )
    st.write(
        f"Dataset: **{metrics['n']} records**, **{metrics['n_features']} predictors**. "
        f"The displayed evaluation uses a stratified 80/20 split (random state {RANDOM_STATE}): "
        f"{metrics['train_n']} training records and {metrics['test_n']} test records. "
        "The interactive demo model is fitted on all dataset records after holdout evaluation."
    )

st.subheader("Model evaluation on the held-out test split")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
c2.metric("ROC-AUC", f"{metrics['auc']:.4f}")
c3.metric("Test records", str(metrics["test_n"]))
c4.metric("Dataset records", str(metrics["n"]))

cm = metrics["cm"]
cm_df = pd.DataFrame(
    cm,
    index=["Actual: No Disease", "Actual: Disease"],
    columns=["Predicted: No Disease", "Predicted: Disease"],
)
st.write("**Confusion matrix**")
st.dataframe(cm_df, use_container_width=True)
report = metrics["report"]
report_df = pd.DataFrame({
    "Class": ["No Disease", "Disease"],
    "Precision": [report["No Disease"]["precision"], report["Disease"]["precision"]],
    "Recall": [report["No Disease"]["recall"], report["Disease"]["recall"]],
    "F1-score": [report["No Disease"]["f1-score"], report["Disease"]["f1-score"]],
})
st.dataframe(report_df.style.format({
    "Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"
}), hide_index=True, use_container_width=True)

st.divider()
st.subheader("Try an example input")
st.write(
    "Enter values using the dataset's original coding. These controls are provided to demonstrate "
    "how a trained classifier accepts features; they are not a health questionnaire."
)

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        age = st.slider("Age (years)", 20, 90, 54)
        sex_label = st.selectbox("Sex code", ["Female (0)", "Male (1)"])
        cp_label = st.selectbox(
            "Chest pain type",
            [
                "1 — Typical angina",
                "2 — Atypical angina",
                "3 — Non-anginal pain",
                "4 — Asymptomatic",
            ],
            index=3,
        )
        trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 220, 130)
        chol = st.slider("Serum cholesterol (mg/dl)", 100, 600, 246)
        fbs_label = st.selectbox("Fasting blood sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
        restecg_label = st.selectbox(
            "Resting ECG result",
            ["0 — Normal", "1 — ST-T wave abnormality", "2 — Left ventricular hypertrophy"],
        )
    with right:
        thalach = st.slider("Maximum heart rate achieved", 60, 220, 150)
        exang_label = st.selectbox("Exercise-induced angina", ["No (0)", "Yes (1)"])
        oldpeak = st.slider("ST depression induced by exercise", 0.0, 6.2, 1.0, 0.1)
        slope_label = st.selectbox(
            "Slope of peak exercise ST segment",
            ["1 — Upsloping", "2 — Flat", "3 — Downsloping"],
            index=1,
        )
        ca = st.selectbox("Number of major vessels (0–3)", [0, 1, 2, 3])
        thal_label = st.selectbox(
            "Thalassemia code",
            ["3 — Normal", "6 — Fixed defect", "7 — Reversible defect"],
        )
    submitted = st.form_submit_button("Run educational prediction", type="primary")

if submitted:
    cp = int(cp_label[0])
    sex = 1 if sex_label.startswith("Male") else 0
    fbs = 1 if fbs_label.startswith("Yes") else 0
    restecg = int(restecg_label[0])
    exang = 1 if exang_label.startswith("Yes") else 0
    slope = int(slope_label[0])
    thal = int(thal_label[0])
    sample = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }], columns=FEATURES)
    probability = float(prediction_model.predict_proba(sample)[0, 1])
    predicted_class = int(probability >= 0.5)
    st.subheader("Model output")
    if predicted_class == 1:
        st.error("The model classifies this input as: Disease (class 1)")
    else:
        st.success("The model classifies this input as: No Disease (class 0)")
    st.metric("Model-estimated probability for class 1", f"{probability * 100:.1f}%")
    st.caption(
        "This probability is an output of a model trained on this historical dataset. "
        "It is not a medically calibrated personal risk estimate or a diagnosis."
    )

st.divider()
st.caption(
    "Project for educational and research demonstration. Dataset: UCI Machine Learning Repository, "
    "Heart Disease (Cleveland subset). No data are sent to an external prediction API."
)
