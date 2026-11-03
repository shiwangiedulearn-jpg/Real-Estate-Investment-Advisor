# Real Estate Investment Advisor

An end-to-end ML project for real-estate investment analysis.

## Tasks
- Regression: predict estimated property price after 5 years.
- Classification: classify whether a property is a Good Investment.

## Final models

### Regression
- Model: Ridge Regression
- MAE: approximately 72,137
- RMSE: approximately 86,560
- R²: approximately 0.86

### Classification
- Model: XGBoost Classifier
- Good Investment target: 5-year appreciation >= 30%
- Accuracy: 0.63
- Precision: 0.6795
- Recall: 0.8154
- F1: 0.7413
- F1 Macro: 0.5461
- Balanced Accuracy: 0.5505
- Confusion matrix: [[10, 25], [12, 53]]

The classifier performs better on the positive class than the negative class, so predictions should not be treated as guaranteed investment outcomes.

## Project structure

```text
Real Estate Investment Advisor/
├── data/
│   ├── data.xlsx
│   └── cleaned_data.xlsx
├── models/
│   ├── ridge_model.pkl
│   ├── scaler.pkl
│   ├── xgb_classifier.pkl
│   └── scaler_cls.pkl
├── notebooks/
│   ├── clean_data.ipynb
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── mlflow_tracking.ipynb
├── mlflow.db
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Run the app

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The `models/` folder must remain beside `streamlit_app.py`.

## MLflow

MLflow tracking uses the local SQLite database `mlflow.db`. The MLflow notebook records the final Ridge and XGBoost runs.

## Streamlit workflow

User input -> feature engineering -> Ridge regression for 5-year price + XGBoost classification for investment category -> results.

## Limitations

The dataset contains 500 records. The investment target is derived from the 5-year appreciation in this dataset. The model is an analytical project and should not be considered financial advice.
