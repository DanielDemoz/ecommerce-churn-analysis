"""
Create demo data for the E-commerce Churn Analysis Dashboard
This script generates sample data if the original dataset is not available.
"""

import pandas as pd
import numpy as np

def create_demo_data():
    """Create demo e-commerce churn dataset"""
    np.random.seed(42)
    n_customers = 1000
    
    # Generate demo data
    data = {
        'CustomerID': range(50001, 50001 + n_customers),
        'Churn': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
        'Tenure': np.random.exponential(12, n_customers).astype(int),
        'PreferredLoginDevice': np.random.choice(['Mobile Phone', 'Computer', 'Phone'], n_customers, p=[0.5, 0.3, 0.2]),
        'CityTier': np.random.choice([1, 2, 3], n_customers, p=[0.4, 0.4, 0.2]),
        'WarehouseToHome': np.random.normal(15, 8, n_customers).astype(int),
        'PreferredPaymentMode': np.random.choice(['Debit Card', 'Credit Card', 'Cash on Delivery', 'E wallet', 'UPI'], n_customers, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
        'Gender': np.random.choice(['Male', 'Female'], n_customers, p=[0.6, 0.4]),
        'HourSpendOnApp': np.random.choice([1, 2, 3, 4, 5], n_customers, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'NumberOfDeviceRegistered': np.random.choice([1, 2, 3, 4, 5, 6], n_customers, p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05]),
        'PreferedOrderCat': np.random.choice(['Laptop & Accessory', 'Mobile', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'], n_customers, p=[0.35, 0.2, 0.15, 0.15, 0.1, 0.05]),
        'SatisfactionScore': np.random.choice([1, 2, 3, 4, 5], n_customers, p=[0.1, 0.15, 0.3, 0.3, 0.15]),
        'MaritalStatus': np.random.choice(['Married', 'Single', 'Divorced'], n_customers, p=[0.55, 0.35, 0.1]),
        'NumberOfAddress': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_customers, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.02, 0.005, 0.005]),
        'Complain': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
        'OrderAmountHikeFromlastYear': np.random.normal(15, 3, n_customers).astype(int),
        'CouponUsed': np.random.choice([0, 1, 2, 3, 4, 5], n_customers, p=[0.2, 0.3, 0.25, 0.15, 0.08, 0.02]),
        'OrderCount': np.random.poisson(8, n_customers),
        'DaySinceLastOrder': np.random.exponential(10, n_customers).astype(int),
        'CashbackAmount': np.random.normal(20, 10, n_customers).astype(int)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    # Customers with low satisfaction are more likely to churn
    df.loc[df['SatisfactionScore'] <= 2, 'Churn'] = np.random.choice([0, 1], len(df[df['SatisfactionScore'] <= 2]), p=[0.3, 0.7])
    
    # Customers with complaints are more likely to churn
    df.loc[df['Complain'] == 1, 'Churn'] = np.random.choice([0, 1], len(df[df['Complain'] == 1]), p=[0.4, 0.6])
    
    # Customers with short tenure are more likely to churn
    df.loc[df['Tenure'] <= 6, 'Churn'] = np.random.choice([0, 1], len(df[df['Tenure'] <= 6]), p=[0.5, 0.5])
    
    # Ensure positive values for certain columns
    df['WarehouseToHome'] = np.maximum(df['WarehouseToHome'], 5)
    df['OrderAmountHikeFromlastYear'] = np.maximum(df['OrderAmountHikeFromlastYear'], 10)
    df['CashbackAmount'] = np.maximum(df['CashbackAmount'], 0)
    df['DaySinceLastOrder'] = np.maximum(df['DaySinceLastOrder'], 1)
    
    return df

if __name__ == "__main__":
    print("Creating demo dataset...")
    demo_df = create_demo_data()
    
    # Save as both Excel and CSV
    demo_df.to_excel("E Commerce Dataset.xlsx", index=False)
    demo_df.to_csv("E Commerce Dataset.csv", index=False)
    
    print(f"✅ Demo dataset created with {len(demo_df)} customers")
    print(f"📊 Churn rate: {demo_df['Churn'].mean():.1%}")
    print("📁 Files saved: E Commerce Dataset.xlsx and E Commerce Dataset.csv")
    print("🚀 You can now run the dashboard!")
