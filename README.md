# E-commerce Churn Dashboard

A Streamlit dashboard for exploring e-commerce churn, comparing two models and scoring a single customer.

**Project page:** https://danieldemoz.github.io/ecommerce-churn-analysis/
**Streamlit app:** https://ecommerce-churn-analysis-hmpsaq7em9psjb8ivlzfzy.streamlit.app/
**Status:** working app. The real dataset is not in this repository, so the hosted app runs on generated demo data.

## What it does

- Explore churn by tenure, satisfaction, device, payment method and other fields.
- Train random forest and logistic regression, and view accuracy, a confusion matrix and feature importance.
- Score one customer profile.

## Data

The app reads `E Commerce Dataset.xlsx` (or `.csv`) from the repository folder. When the file is missing, it generates 1,000 demo customers with churn patterns built in: low satisfaction, complaints and short tenure raise churn.

On that demo data (`analysis/demo_check.py`, 5-fold cross-validation):

| Model | Accuracy | AUC |
|---|---|---|
| Always guess "stays" | 56.7% | 0.500 |
| Logistic regression | 56.8% | 0.573 |
| Random forest | 58.8% | 0.631 |

**Honest finding:** the models barely beat guessing on demo data, and the "churn drivers" they find are the generator's own rules. The 95.8% accuracy and driver list in the earlier README could not be reproduced from this repository and were removed.

## Responsible AI, following the Reliable AI Adoption Framework

1. **Business problem:** spot customers likely to stop buying, so a retention team can reach them first.
2. **Data readiness:** the real dataset is not included. The hosted app runs on generated demo data with planted patterns.
3. **Governance and risk:** scores from demo data must never be presented as customer insight.
4. **Technology and tools:** random forest and logistic regression in Streamlit, simple and inspectable.
5. **People and adoption:** a retention lead reviews the ranked list and chooses the offer.
6. **Outcomes and monitoring:** on real data, compare predicted and actual churn each quarter, by customer group.

## How to run

```bash
pip install -r requirements.txt
streamlit run dashboard.py
python analysis/demo_check.py     # scores the models on the demo data
pytest -q
```

## Tech stack

Python, Streamlit, pandas, scikit-learn, Plotly.
