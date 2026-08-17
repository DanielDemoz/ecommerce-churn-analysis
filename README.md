# E-commerce Churn Analysis Dashboard

Streamlit dashboard for predicting e-commerce customer churn, segmenting at-risk users, and surfacing retention insights.

## Problem

E-commerce platforms lose revenue when customers churn silently; teams need a practical way to score churn risk, compare models, and identify actionable retention levers.

## Approach

Built an interactive Streamlit app over e-commerce customer data (demographics, behavior, transactions, satisfaction). Trained Random Forest and Logistic Regression classifiers, added customer segmentation views, and wired a prediction page for single-customer risk scoring with demo data fallback when no CSV is uploaded.

## Results

- Random Forest: 95.8% accuracy; Logistic Regression: 84.0%
- Churn precision: 99.3%; recall: 75.1%
- Key churn indicators: low satisfaction (<3/5), short tenure (<6 months), low app usage (<2 hours), multiple addresses, prior complaints

## Tech stack

Python, Streamlit, scikit-learn, XGBoost, pandas, NumPy, Plotly, matplotlib, seaborn

## How to run

```bash
git clone https://github.com/DanielDemoz/ecommerce-churn-analysis.git
cd ecommerce-churn-analysis
pip install -r requirements.txt
streamlit run dashboard.py
```

Open http://localhost:8501

## Screenshot / demo

**Live dashboard:** https://ecommerce-churn-analysis-hmpsaq7em9psjb8ivlzfzy.streamlit.app/
