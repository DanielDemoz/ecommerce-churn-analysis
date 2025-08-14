# E-commerce Churn Analysis

## Project Overview
This project focuses on **Customer Churn Prediction & Segmentation** for an e-commerce platform using **Machine Learning**.  
The goal is to identify customers at high risk of leaving and provide actionable insights for retention strategies.

By leveraging **data preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling**, this project demonstrates how data-driven decisions can improve customer retention and business profitability.

---

## Dataset
**File:** `E Commerce Dataset.xlsx`  
The dataset contains customer attributes such as:
- Demographics (City Tier, etc.)
- Usage patterns (Hours Spent on App, Website Visits)
- Satisfaction scores
- Purchase frequency & monetary value

---

## Key Steps in the Project

### 1. **Data Cleaning & Preparation**
- Handled missing values and outliers (see `comparison of outlier before and after handling.png`).
- Converted categorical variables to numerical using encoding techniques.
- Created new features for customer segmentation.

### 2. **Exploratory Data Analysis (EDA)**
- Correlation analysis (`Correlation Matrix.png`).
- Outlier detection (`Outliers detection.png`).
- Customer satisfaction analysis (`Satisfaction Score Distribution by City Tier.png`).
- Usage behaviour insights (`Hours Spent on App.png`).

### 3. **Feature Engineering**
- Normalization & scaling of numerical features.
- Derived metrics for **RFM analysis** (Recency, Frequency, Monetary Value).

### 4. **Machine Learning Modeling**
- Built churn prediction models using:
  - **Logistic Regression**
  - **Decision Trees**
  - **Random Forest**
- Evaluated models using **accuracy, precision, recall, and F1-score**.
- Selected the best-performing model for deployment.

---

## Results & Insights
- Identified top churn indicators:
  - Low satisfaction scores
  - Low purchase frequency
  - Short time spent on app
- Recommended retention strategies:
  - Personalized offers for high-risk customers
  - Engagement campaigns for low-usage customers
  - Service improvements for dissatisfied customers

---

## Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Tools:** Jupyter Notebook
- **Data Visualization:** Matplotlib, Seaborn

---

## Repository Structure


