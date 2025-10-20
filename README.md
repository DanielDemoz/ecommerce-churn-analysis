# E-commerce Churn Analysis Dashboard

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit%20Cloud-blue?style=for-the-badge&logo=streamlit)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)

## Interactive Dashboard Access

**Deploy to Streamlit Cloud:** [Deploy Now](https://share.streamlit.io)

*Deploy this dashboard to Streamlit Cloud for free and get a live URL. The dashboard includes real-time data visualization, machine learning predictions, and customer segmentation analysis.*

**Local Testing:** Run `streamlit run dashboard.py` and visit http://localhost:8501

---

## Project Overview

This project provides a comprehensive **Interactive Dashboard** for **Customer Churn Prediction & Segmentation** in e-commerce platforms using **Machine Learning**. The dashboard offers real-time insights, predictive analytics, and actionable recommendations for customer retention strategies.

### Key Features
- **Interactive Web Dashboard** with easy navigation
- **Real-time Data Visualization** with Plotly charts
- **Machine Learning Models** (Random Forest, Logistic Regression)
- **Customer Churn Prediction** with risk assessment
- **Customer Segmentation** analysis
- **Comprehensive Analytics** and insights

---

## Quick Start

### Option 1: Streamlit Cloud (Recommended)
1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub account
4. Deploy the app with one click

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/DanielDemoz/ecommerce-churn-analysis.git
cd ecommerce-churn-analysis

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard.py
```


---

## Dataset
The dashboard works with e-commerce customer datasets containing the following attributes:
- **Demographics:** City Tier, Gender, Marital Status
- **Behavioral:** Hours Spent on App, Preferred Login Device
- **Transactional:** Order Count, Payment Mode, Order Categories
- **Satisfaction:** Satisfaction Score, Complaints
- **Target:** Churn Status (0/1)

*Note: The dashboard will automatically create demo data if no dataset is provided.*

---

## Dashboard Features

### Overview Page
- Key metrics and KPIs
- Dataset summary
- Quick insights and trends
- Churn distribution visualization

### Data Analysis Page
- **Distribution Analysis:** Histograms and statistical distributions
- **Correlation Analysis:** Feature correlation heatmaps
- **Churn Factors:** Categorical analysis and risk factors
- **Customer Behavior:** Usage patterns and satisfaction analysis

### Model Performance Page
- Model comparison (Random Forest vs Logistic Regression)
- Accuracy metrics and performance indicators
- Confusion matrices
- Feature importance rankings

### Customer Segmentation Page
- Tenure-based customer segments
- Satisfaction-based segments
- Segment-wise churn analysis
- Behavioral clustering insights

### Predictions Page
- Interactive customer churn prediction
- Risk assessment with probability scores
- Real-time model predictions
- Actionable recommendations

---

## Technical Stack

### Backend & Analytics
- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **XGBoost** - Advanced ML models

### Visualization & Dashboard
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical visualizations

### Data Processing
- **OpenPyXL** - Excel file handling
- **SciPy** - Scientific computing
- **Statsmodels** - Statistical modeling

---

## Key Insights & Results

### Churn Indicators Identified:
- **Low satisfaction scores** (below 3/5)
- **Short tenure** (less than 6 months)
- **Low app usage** (less than 2 hours)
- **Multiple addresses** (indicates instability)
- **Previous complaints** (high correlation)

### Model Performance:
- **Random Forest:** 95.8% accuracy
- **Logistic Regression:** 84.0% accuracy
- **Precision:** 99.3% for churn prediction
- **Recall:** 75.1% for churn detection

### Retention Strategies:
- **Personalized offers** for high-risk customers
- **Engagement campaigns** for low-usage customers
- **Service improvements** for dissatisfied customers
- **Loyalty programs** for long-tenure customers

---

## Configuration

### Dashboard Settings (`config.toml`)
- Port: 8501
- Theme: Light mode with custom colors
- CORS: Disabled for local development
- Upload limit: 200MB

### Model Parameters
- **Random Forest:** 100 estimators, random state 42
- **Logistic Regression:** Max iterations 1000
- **Train/Test Split:** 80/20 ratio
- **Cross-validation:** 5-fold

---

## Usage Instructions

### For Business Users:
1. **Navigate** through different pages using the sidebar
2. **Explore** data insights and trends
3. **Use predictions** to assess customer risk
4. **Analyze segments** for targeted campaigns

### For Data Scientists:
1. **Review model performance** metrics
2. **Examine feature importance** rankings
3. **Validate predictions** with test data
4. **Customize models** as needed

---

## Deployment Options

### Local Development
```bash
python run_dashboard.py
```

### Production Deployment
```bash
# Using Streamlit Cloud
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Future Enhancements

- **Real-time data integration**
- **Email alerts** for high-risk customers
- **A/B testing** for retention campaigns
- **Advanced analytics** with time series
- **User authentication** and role-based access
- **Mobile-responsive** design improvements

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Support

For questions or support:
- **Email:** [your-email@domain.com]
- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation:** [Wiki](https://github.com/your-repo/wiki)

---

**Happy Analyzing! Make data-driven decisions to reduce customer churn and boost retention!**


