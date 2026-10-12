# Heart Disease Likelihood Prediction — Logistic Regression

An educational Streamlit app built with Logistic Regression and the bundled UCI Cleveland Heart Disease dataset.

## Run locally

1. Extract this folder.
2. Open the folder in VS Code.
3. In the terminal, install packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

The dataset is included at `data/processed.cleveland.data`; the app does not fetch it at runtime. Package installation may require internet access the first time.

## Deploy to Streamlit Community Cloud

1. Upload the contents of this folder to a GitHub repository (keep the `data` folder).
2. Visit https://share.streamlit.io and sign in with GitHub.
3. Select **Create app**.
4. Choose the repository, branch `main`, and main file path `app.py`.
5. Click **Deploy**.

The app's required dependencies are listed in `requirements.txt`. Streamlit Community Cloud runs the app from the repository and needs locally used files to be included in the repository.

## Method

- Target mapping: original `num=0` -> class 0 (No Disease); original `num=1,2,3,4` -> class 1 (Disease).
- Missing predictor values: median imputation inside the scikit-learn pipeline.
- Scaling: StandardScaler.
- Classifier: LogisticRegression(max_iter=1000, random_state=42).
- Evaluation: stratified 80/20 split, random_state=42.
- Evaluation metrics are calculated on the held-out split. The interactive demonstration model is fitted separately on all available rows after evaluation.

## Important limitation

This is an academic demonstration on a small historical dataset. It is not a diagnostic tool, not a clinical risk calculator, and not suitable for healthcare decisions. The test split is internal and does not establish performance on new populations.
