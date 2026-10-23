from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "best_churn_model.joblib"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

for path in [DATA, MODEL_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing: {path}\n"
            "Run scripts/check_data.py and scripts/train_models.py first."
        )

df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["Churn"] = df["Churn"].astype(str).str.strip()
df = df[df["Churn"].isin(["Yes", "No"])].copy()
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

model = joblib.load(MODEL_PATH)

# Match the training features
X = df.drop(columns=["Churn", "ChurnFlag"], errors="ignore")
X = X.drop(columns=["customerID"], errors="ignore")

# 1. Overall churn distribution
counts = df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(counts.index, counts.values)
ax.set_title("Customer Churn Distribution")
ax.set_xlabel("Churn")
ax.set_ylabel("Number of Customers")
for i, value in enumerate(counts.values):
    ax.text(i, value, str(value), ha="center", va="bottom")
fig.tight_layout()
fig.savefig(FIG_DIR / "churn_distribution.png", dpi=160)
plt.close(fig)

# 2. Churn rate by contract type
if "Contract" in df.columns:
    rates = (
        df.groupby("Contract", observed=False)["ChurnFlag"]
        .mean().sort_values(ascending=False) * 100
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    rates.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_by_contract.png", dpi=160)
    plt.close(fig)

    rates.rename("churn_rate_percent").to_csv(
        TABLE_DIR / "churn_rate_by_contract.csv"
    )

# 3. Churn rate by tenure
if "tenure" in df.columns:
    bins = [-1, 6, 12, 24, 48, 72]
    labels = ["0-6", "7-12", "13-24", "25-48", "49-72"]
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=bins, labels=labels
    )
    rates = (
        df.groupby("TenureGroup", observed=False)["ChurnFlag"].mean() * 100
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    rates.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Customer Tenure")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_by_tenure.png", dpi=160)
    plt.close(fig)

    rates.rename("churn_rate_percent").to_csv(
        TABLE_DIR / "churn_rate_by_tenure.csv"
    )

# 4. Monthly charges by churn status
if "MonthlyCharges" in df.columns:
    groups = [
        df.loc[df["Churn"] == "No", "MonthlyCharges"].dropna(),
        df.loc[df["Churn"] == "Yes", "MonthlyCharges"].dropna()
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(groups, tick_labels=["No Churn", "Churn"])
    ax.set_title("Monthly Charges by Churn Status")
    ax.set_ylabel("Monthly Charges")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_charges_by_churn.png", dpi=160)
    plt.close(fig)

# 5. Predict churn risk for all available customers
probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

risk = pd.DataFrame({
    "customerID": (
        df["customerID"].values
        if "customerID" in df.columns
        else np.arange(len(df))
    ),
    "actual_churn": df["Churn"].values,
    "predicted_churn_probability": probabilities,
    "predicted_churn_at_0_5_threshold": predictions
})

risk["risk_category"] = pd.cut(
    risk["predicted_churn_probability"],
    bins=[-0.001, 0.30, 0.60, 1.001],
    labels=["Low", "Medium", "High"]
)

risk = risk.sort_values(
    "predicted_churn_probability", ascending=False
)
risk.to_csv(TABLE_DIR / "customer_churn_risk.csv", index=False)

# 6. Model feature importance / coefficients
pipeline = model
preprocessor = pipeline.named_steps["preprocessor"]
estimator = pipeline.named_steps["model"]

try:
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance_label = "importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
        importance_label = "absolute_coefficient"
    else:
        values = None

    if values is not None and len(feature_names) == len(values):
        importance = pd.DataFrame({
            "feature": feature_names,
            importance_label: values
        }).sort_values(importance_label, ascending=False)

        importance.to_csv(
            TABLE_DIR / "feature_importance.csv", index=False
        )

        top = importance.head(15).sort_values(importance_label)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top["feature"], top[importance_label])
        ax.set_title("Top 15 Model Features")
        ax.set_xlabel(importance_label.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(FIG_DIR / "top_feature_importance.png", dpi=160)
        plt.close(fig)
except Exception as exc:
    print("Feature importance could not be generated:", exc)

# 7. Summary table
summary = {
    "customers_analyzed": int(len(df)),
    "actual_churn_rate_percent": round(
        float(df["ChurnFlag"].mean() * 100), 2
    ),
    "predicted_high_risk_customers": int(
        (risk["risk_category"] == "High").sum()
    ),
    "predicted_medium_risk_customers": int(
        (risk["risk_category"] == "Medium").sum()
    ),
    "predicted_low_risk_customers": int(
        (risk["risk_category"] == "Low").sum()
    )
}

pd.DataFrame([summary]).to_csv(
    TABLE_DIR / "analysis_summary.csv", index=False
)

print("\nANALYSIS COMPLETED")
print(pd.Series(summary).to_string())
print("\nCharts saved to:", FIG_DIR)
print("Tables saved to:", TABLE_DIR)
print("\nGenerated charts:")
for path in sorted(FIG_DIR.glob("*.png")):
    print(" -", path.name)
