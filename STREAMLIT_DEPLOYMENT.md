# Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### 1. Go to Streamlit Cloud
Visit: https://share.streamlit.io/

### 2. Sign In
- Click "Sign in with GitHub"
- Authorize Streamlit to access your GitHub account

### 3. Deploy Your App
- Click "New app"
- **Repository:** `DanielDemoz/ecommerce-churn-analysis`
- **Branch:** `main`
- **Main file path:** `dashboard.py`
- **App URL:** `ecommerce-churn-analysis` (or your preferred name)

### 4. Wait for Deployment
- Streamlit will automatically install dependencies from `requirements.txt`
- Deployment takes 2-5 minutes
- You'll get a URL like: `https://ecommerce-churn-analysis.streamlit.app`

## After Deployment

Once deployed, you can:
1. Update the README.md with your actual dashboard URL
2. Share the live dashboard with others
3. The dashboard will auto-update when you push changes to GitHub

## Troubleshooting

### Common Issues:
- **Import errors:** Check that all dependencies are in `requirements.txt`
- **File not found:** Ensure `dashboard.py` is in the root directory
- **Data loading issues:** The dashboard will create demo data if no dataset is found

### Support:
- Streamlit Cloud Documentation: https://docs.streamlit.io/streamlit-community-cloud
- GitHub Issues: Create an issue in your repository

## Local Testing

To test before deploying:
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Then visit: http://localhost:8501
