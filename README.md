# Customer Churn Prediction & Retention Analytics

An end-to-end machine learning project to predict customer churn
and support data-driven customer retention decisions.

## Project Objectives

- Explore customer behavior and churn patterns.
- Build and compare classification models.
- Evaluate performance using precision, recall, F1-score and ROC-AUC.
- Identify high-risk customers for potential retention campaigns.
- Produce reproducible analysis charts and CSV reports.

## Dataset

IBM Telco Customer Churn dataset.

Source:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Target: `Churn` (Yes/No).

## Models

- Logistic Regression
- Random Forest

Models are compared using a stratified train/test split. The selected model
is saved locally for reuse.

## Project Structure

```text
customer-churn-prediction/
├── data/raw/
├── models/
├── reports/
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── check_data.py
│   ├── train_models.py
│   └── analyze_churn.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

Place the dataset at:

`data/raw/Telco-Customer-Churn.csv`

## Run the Pipeline

```bash
python scripts/check_data.py
python scripts/train_models.py
python scripts/analyze_churn.py
```

## Outputs

- `models/best_churn_model.joblib`
- `reports/tables/model_comparison.csv`
- `reports/tables/customer_churn_risk.csv`
- `reports/tables/feature_importance.csv`
- Charts in `reports/figures/`

## Evaluation Notes

The reported test metrics should be used to compare models. The risk scores
generated for the full dataset are illustrative and should not be treated as
independent test results.

Feature importance indicates model associations, not causal effects.
Customer risk predictions should be validated before operational use.

## Author

Economics, econometrics and data science.

## License

Add a license before redistributing this project or dataset.
