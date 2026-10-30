"""Score the app's models on the demo data it generates, and write docs/check.json.

The app needs "E Commerce Dataset.xlsx", which is not in this repository. Without it,
dashboard.py generates 1,000 demo customers with churn rules built in. This script
shows what a model learns from that demo data.

Run from the repo root:  python analysis/demo_check.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "dashboard.py").read_text()
start = src.index("def create_demo_data_inline")
end = src.index("@st.cache_data", start)
ns = {"np": np, "pd": pd}
exec(src[start:end], ns)  # the generator exactly as the app runs it
df = ns["create_demo_data_inline"]()

y = df.Churn
X = pd.get_dummies(df.drop(columns=["Churn", "CustomerID"], errors="ignore"), drop_first=True, dtype=float)
cv = StratifiedKFold(5, shuffle=True, random_state=42)


def acc(model):
    return round(float(cross_val_score(model, X, y, cv=cv).mean()), 3)


def auc(model):
    return round(float(cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()), 3)


rf = RandomForestClassifier(n_estimators=300, random_state=0)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
planted = (df.SatisfactionScore <= 2) | (df.Complain == 1) | (df.Tenure <= 6)
out = {
    "rows": int(len(df)),
    "churn_rate": round(float(y.mean()), 3),
    "majority_accuracy": round(float(max(y.mean(), 1 - y.mean())), 3),
    "random_forest": {"accuracy": acc(rf), "auc": auc(rf)},
    "logistic_regression": {"accuracy": acc(lr), "auc": auc(lr)},
    "planted_rule": {
        "description": "satisfaction 2 or lower, a complaint, or tenure of 6 months or less",
        "share_of_customers": round(float(planted.mean()), 3),
        "churn_if_rule": round(float(y[planted].mean()), 3),
        "churn_if_not": round(float(y[~planted].mean()), 3),
    },
}
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
