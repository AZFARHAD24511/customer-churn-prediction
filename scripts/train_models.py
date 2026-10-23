from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "tables"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA}")

df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()

target = "Churn"
if target not in df.columns:
    raise ValueError(f"Target '{target}' not found in dataset.")

# Clean common issues in the Telco dataset
df = df.drop_duplicates().copy()
df[target] = df[target].astype(str).str.strip()
df = df[df[target].isin(["Yes", "No"])].copy()
y = df[target].map({"No": 0, "Yes": 1})

X = df.drop(columns=[target])

# Customer ID is an identifier, not a useful predictive feature
if "customerID" in X.columns:
    X = X.drop(columns=["customerID"])

# Convert TotalCharges to numeric; blank values become missing
if "TotalCharges" in X.columns:
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
}

results = []
best_auc = -1
best_name = None
best_pipeline = None

print(f"\nDataset: {df.shape[0]} rows, {X.shape[1]} features")
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
print(f"Churn rate: {y.mean():.2%}\n")

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    print(f"Training {name}...")
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob)
    }
    results.append(metrics)

    print(pd.Series(metrics).to_string())
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(
        y_test, pred, target_names=["No Churn", "Churn"], zero_division=0
    ))
    print("-" * 55)

    if metrics["roc_auc"] > best_auc:
        best_auc = metrics["roc_auc"]
        best_name = name
        best_pipeline = pipeline

results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
best_pipeline_path = MODEL_DIR / "best_churn_model.joblib"
joblib.dump(best_pipeline, best_pipeline_path)

with open(REPORT_DIR / "best_model.json", "w", encoding="utf-8") as f:
    json.dump(
        {"best_model": best_name, "test_roc_auc": float(best_auc)},
        f, indent=2
    )

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print(f"\nBest model: {best_name} | ROC-AUC: {best_auc:.4f}")
print(f"Saved model: {best_pipeline_path}")
print(f"Saved metrics: {REPORT_DIR / 'model_comparison.csv'}")
