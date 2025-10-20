import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="E-commerce Churn Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the e-commerce dataset"""
    try:
        # Try to load from Excel file
        df = pd.read_excel("E Commerce Dataset.xlsx")
    except:
        # If Excel fails, try CSV
        try:
            df = pd.read_csv("E Commerce Dataset.csv")
        except:
            # If no dataset is available, create demo data
            st.warning("⚠️ No dataset found. Creating demo data for demonstration purposes...")
            try:
                import subprocess
                subprocess.run(["python", "create_demo_data.py"], check=True)
                df = pd.read_excel("E Commerce Dataset.xlsx")
                st.success("✅ Demo dataset created successfully!")
            except:
                st.error("❌ Could not create demo data. Please ensure the dataset file is available.")
                return None
    
    # Data preprocessing
    # Handle missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

@st.cache_data
def get_model_predictions(df):
    """Train models and get predictions"""
    # Prepare features
    X = df.drop(['Churn', 'CustomerID'], axis=1, errors='ignore')
    y = df['Churn']
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    return {
        'rf_model': rf_model,
        'lr_model': lr_model,
        'rf_accuracy': rf_accuracy,
        'lr_accuracy': lr_accuracy,
        'rf_pred': rf_pred,
        'lr_pred': lr_pred,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'label_encoders': le_dict
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 E-commerce Churn Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "📈 Data Analysis", "🤖 Model Performance", "👥 Customer Segmentation", "🔮 Predictions"]
    )
    
    # Main content based on selected page
    if page == "🏠 Overview":
        show_overview(df)
    elif page == "📈 Data Analysis":
        show_data_analysis(df)
    elif page == "🤖 Model Performance":
        show_model_performance(df)
    elif page == "👥 Customer Segmentation":
        show_customer_segmentation(df)
    elif page == "🔮 Predictions":
        show_predictions(df)

def show_overview(df):
    """Display overview metrics and key insights"""
    st.header("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        churn_rate = df['Churn'].mean() * 100
        st.metric(
            label="Churn Rate",
            value=f"{churn_rate:.1f}%",
            delta=None
        )
    
    with col3:
        avg_tenure = df['Tenure'].mean()
        st.metric(
            label="Avg Tenure (months)",
            value=f"{avg_tenure:.1f}",
            delta=None
        )
    
    with col4:
        avg_satisfaction = df['SatisfactionScore'].mean()
        st.metric(
            label="Avg Satisfaction",
            value=f"{avg_satisfaction:.1f}/5",
            delta=None
        )
    
    # Dataset info
    st.subheader("📋 Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Columns:**", len(df.columns))
        st.write("**Missing Values:**", df.isnull().sum().sum())
    
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes.value_counts())
    
    # Quick insights
    st.subheader("🔍 Quick Insights")
    
    # Churn distribution
    fig = px.pie(
        values=df['Churn'].value_counts().values,
        names=['Retained', 'Churned'],
        title="Customer Churn Distribution",
        color_discrete_sequence=['#2E8B57', '#DC143C']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top insights
    st.subheader("💡 Key Insights")
    
    insights = [
        f"• **{churn_rate:.1f}%** of customers have churned",
        f"• Average customer tenure is **{avg_tenure:.1f} months**",
        f"• Average satisfaction score is **{avg_satisfaction:.1f}/5**",
        f"• **{df['Gender'].value_counts().index[0]}** customers make up the majority",
        f"• **{df['PreferredPaymentMode'].value_counts().index[0]}** is the most preferred payment method"
    ]
    
    for insight in insights:
        st.markdown(insight)

def show_data_analysis(df):
    """Display detailed data analysis and visualizations"""
    st.header("📈 Data Analysis & Visualizations")
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Distribution Analysis", "Correlation Analysis", "Churn Factors", "Customer Behavior"]
    )
    
    if analysis_type == "Distribution Analysis":
        show_distribution_analysis(df)
    elif analysis_type == "Correlation Analysis":
        show_correlation_analysis(df)
    elif analysis_type == "Churn Factors":
        show_churn_factors(df)
    elif analysis_type == "Customer Behavior":
        show_customer_behavior(df)

def show_distribution_analysis(df):
    """Show distribution analysis"""
    st.subheader("📊 Distribution Analysis")
    
    # Select columns to analyze
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect(
        "Select columns to analyze:",
        numerical_cols,
        default=numerical_cols[:4]
    )
    
    if selected_cols:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=selected_cols[:4],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, col in enumerate(selected_cols[:4]):
            row = (i // 2) + 1
            col_pos = (i % 2) + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, nbinsx=30),
                row=row, col=col_pos
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_correlation_analysis(df):
    """Show correlation analysis"""
    st.subheader("🔗 Correlation Analysis")
    
    # Select numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Plot correlation heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations with Churn
    churn_corr = corr_matrix['Churn'].abs().sort_values(ascending=False)
    st.subheader("Top Correlations with Churn")
    st.bar_chart(churn_corr[1:])  # Exclude Churn itself

def show_churn_factors(df):
    """Show churn factor analysis"""
    st.subheader("🎯 Churn Factor Analysis")
    
    # Categorical analysis
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        selected_cat = st.selectbox("Select categorical variable:", categorical_cols)
        
        # Churn rate by category
        churn_by_cat = df.groupby(selected_cat)['Churn'].agg(['count', 'sum', 'mean']).reset_index()
        churn_by_cat.columns = [selected_cat, 'Total', 'Churned', 'Churn_Rate']
        churn_by_cat['Churn_Rate'] = churn_by_cat['Churn_Rate'] * 100
        
        # Plot
        fig = px.bar(
            churn_by_cat,
            x=selected_cat,
            y='Churn_Rate',
            title=f"Churn Rate by {selected_cat}",
            text='Churn_Rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.dataframe(churn_by_cat)

def show_customer_behavior(df):
    """Show customer behavior analysis"""
    st.subheader("👤 Customer Behavior Analysis")
    
    # Hours spent on app vs churn
    fig = px.box(
        df,
        x='Churn',
        y='HourSpendOnApp',
        title="Hours Spent on App vs Churn Status",
        color='Churn',
        color_discrete_sequence=['#2E8B57', '#DC143C']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Satisfaction score vs churn
    fig = px.box(
        df,
        x='Churn',
        y='SatisfactionScore',
        title="Satisfaction Score vs Churn Status",
        color='Churn',
        color_discrete_sequence=['#2E8B57', '#DC143C']
    )
    st.plotly_chart(fig, use_container_width=True)

def show_model_performance(df):
    """Display model performance metrics"""
    st.header("🤖 Model Performance")
    
    # Get model results
    model_results = get_model_predictions(df)
    
    # Model comparison
    st.subheader("📊 Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Random Forest Accuracy",
            value=f"{model_results['rf_accuracy']:.3f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Logistic Regression Accuracy",
            value=f"{model_results['lr_accuracy']:.3f}",
            delta=None
        )
    
    # Confusion Matrix
    st.subheader("📈 Confusion Matrix - Random Forest")
    
    cm = confusion_matrix(model_results['y_test'], model_results['rf_pred'])
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
        x=['Not Churn', 'Churn'],
        y=['Not Churn', 'Churn']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance - Random Forest")
    
    feature_importance = pd.DataFrame({
        'feature': model_results['feature_names'],
        'importance': model_results['rf_model'].feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_customer_segmentation(df):
    """Display customer segmentation analysis"""
    st.header("👥 Customer Segmentation")
    
    # Create customer segments based on key metrics
    df_seg = df.copy()
    
    # Create segments based on tenure and satisfaction
    df_seg['Tenure_Segment'] = pd.cut(
        df_seg['Tenure'],
        bins=[0, 6, 24, float('inf')],
        labels=['New', 'Established', 'Loyal']
    )
    
    df_seg['Satisfaction_Segment'] = pd.cut(
        df_seg['SatisfactionScore'],
        bins=[0, 2, 4, 5],
        labels=['Low', 'Medium', 'High']
    )
    
    # Segment analysis
    st.subheader("📊 Customer Segments")
    
    # Tenure segments
    tenure_segments = df_seg.groupby('Tenure_Segment').agg({
        'Churn': ['count', 'sum', 'mean'],
        'SatisfactionScore': 'mean',
        'HourSpendOnApp': 'mean'
    }).round(2)
    
    st.write("**Tenure-based Segments:**")
    st.dataframe(tenure_segments)
    
    # Visualization
    fig = px.bar(
        df_seg.groupby('Tenure_Segment')['Churn'].mean().reset_index(),
        x='Tenure_Segment',
        y='Churn',
        title="Churn Rate by Tenure Segment",
        color='Churn',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Satisfaction segments
    fig = px.bar(
        df_seg.groupby('Satisfaction_Segment')['Churn'].mean().reset_index(),
        x='Satisfaction_Segment',
        y='Churn',
        title="Churn Rate by Satisfaction Segment",
        color='Churn',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_predictions(df):
    """Display prediction interface"""
    st.header("🔮 Customer Churn Prediction")
    
    # Get model results
    model_results = get_model_predictions(df)
    
    st.subheader("📝 Enter Customer Information")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (months)", 0, 60, 12)
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        hours_app = st.slider("Hours Spent on App", 0, 5, 2)
        devices = st.slider("Number of Devices", 1, 6, 3)
    
    with col2:
        warehouse_home = st.slider("Warehouse to Home Distance", 5, 50, 15)
        addresses = st.slider("Number of Addresses", 1, 10, 3)
        order_hike = st.slider("Order Amount Hike (%)", 10, 25, 15)
        cashback = st.slider("Cashback Amount", 0, 100, 20)
    
    # Categorical inputs
    col3, col4 = st.columns(2)
    
    with col3:
        login_device = st.selectbox("Preferred Login Device", df['PreferredLoginDevice'].unique())
        payment_mode = st.selectbox("Preferred Payment Mode", df['PreferredPaymentMode'].unique())
    
    with col4:
        gender = st.selectbox("Gender", df['Gender'].unique())
        order_cat = st.selectbox("Preferred Order Category", df['PreferedOrderCat'].unique())
    
    # Predict button
    if st.button("🔮 Predict Churn Risk"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Tenure': [tenure],
            'SatisfactionScore': [satisfaction],
            'HourSpendOnApp': [hours_app],
            'NumberOfDeviceRegistered': [devices],
            'WarehouseToHome': [warehouse_home],
            'NumberOfAddress': [addresses],
            'OrderAmountHikeFromlastYear': [order_hike],
            'CashbackAmount': [cashback],
            'PreferredLoginDevice': [login_device],
            'PreferredPaymentMode': [payment_mode],
            'Gender': [gender],
            'PreferedOrderCat': [order_cat]
        })
        
        # Encode categorical variables
        for col, le in model_results['label_encoders'].items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col].astype(str))
        
        # Make prediction
        rf_pred = model_results['rf_model'].predict(input_data)[0]
        rf_prob = model_results['rf_model'].predict_proba(input_data)[0]
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if rf_pred == 1:
                st.error("🚨 **HIGH CHURN RISK**")
            else:
                st.success("✅ **LOW CHURN RISK**")
        
        with col2:
            st.metric(
                label="Churn Probability",
                value=f"{rf_prob[1]:.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Retention Probability",
                value=f"{rf_prob[0]:.1%}",
                delta=None
            )
        
        # Risk level
        if rf_prob[1] > 0.7:
            st.warning("⚠️ **High Risk**: Immediate intervention recommended")
        elif rf_prob[1] > 0.4:
            st.info("⚠️ **Medium Risk**: Monitor closely and consider retention strategies")
        else:
            st.success("✅ **Low Risk**: Customer appears stable")

if __name__ == "__main__":
    main()
