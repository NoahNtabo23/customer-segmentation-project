"""
Customer Segmentation Dashboard
Simplified version - all files in the app folder
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🛍️",
    layout="wide"
)

# ============================================================================
# LOAD MODEL AND DATA - Using files in the same folder (app/)
# ============================================================================

@st.cache_resource
def load_model_and_scaler():
    """Load the trained K-Means model and scaler from the app folder"""
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(current_dir, 'kmeans_model.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

@st.cache_data
def load_data():
    """Load the dataset from the app folder"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'Mall_Customers.csv')
    df = pd.read_csv(data_path)
    return df

# Load everything
try:
    model, scaler = load_model_and_scaler()
    df = load_data()
    st.success("✅ Model and data loaded successfully!")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.info("Make sure these files are in the app folder:")
    st.info("1. kmeans_model.pkl")
    st.info("2. scaler.pkl")
    st.info("3. Mall_Customers.csv")
    st.stop()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🛍️ Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to:",
    ["📊 Data Overview", "📈 EDA & Visualizations", "🎯 Cluster Prediction", "📊 Business Insights"]
)

# ============================================================================
# PAGE 1: DATA OVERVIEW
# ============================================================================

if page == "📊 Data Overview":
    st.title("📊 Customer Data Overview")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        female_count = len(df[df['Gender'] == 'Female'])
        st.metric("Female Customers", female_count)
    with col4:
        male_count = len(df[df['Gender'] == 'Male'])
        st.metric("Male Customers", male_count)
    
    st.markdown("---")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())

# ============================================================================
# PAGE 2: EDA & VISUALIZATIONS
# ============================================================================

elif page == "📈 EDA & Visualizations":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("---")
    
    # Add cluster labels
    # Scaler was trained on: Age, Income, Spending (3 features)
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    features_scaled = scaler.transform(features)
    
    # Model was trained on Income and Spending (2 features)
    # Use columns 1 and 2 (Income and Spending)
    df['Cluster'] = model.predict(features_scaled[:, 1:3])
    
    cluster_names = {
        0: "High-Value Loyal",
        1: "Savers",
        2: "Spending Beyond Means",
        3: "Low-Value",
        4: "Average"
    }
    df['Segment'] = df['Cluster'].map(cluster_names)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['Age'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title('Age Distribution')
        ax.set_xlabel('Age (years)')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        gender_counts = df['Gender'].value_counts()
        ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
               colors=['pink', 'lightblue'], startangle=90)
        ax.set_title('Gender Distribution')
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Income vs Spending Score")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                    hue='Segment', palette='viridis', s=80, ax=ax)
    ax.set_title('Income vs Spending Score (Customer Segments)')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Customer Segments Overview")
    segment_counts = df['Segment'].value_counts()
    st.dataframe(segment_counts)

# ============================================================================
# PAGE 3: CLUSTER PREDICTION
# ============================================================================

elif page == "🎯 Cluster Prediction":
    st.title("🎯 Predict Customer Segment")
    st.markdown("---")
    
    st.write("Enter customer details to predict their segment:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 18, 70, 30)
    with col2:
        income = st.slider("Annual Income (k$)", 15, 140, 60)
    with col3:
        spending = st.slider("Spending Score (1-100)", 1, 100, 50)
    
    if st.button("🔮 Predict Cluster", type="primary"):
        # Scale using ALL 3 features (Age, Income, Spending)
        input_data = np.array([[age, income, spending]])
        input_scaled = scaler.transform(input_data)
        
        # Use only Income and Spending for prediction
        cluster = model.predict(input_scaled[:, 1:3])[0]
        
        cluster_labels = {
            0: "💰 High-Value Loyal Customer",
            1: "🛡️ Saver / Careful Spender",
            2: "💳 Spending Beyond Means",
            3: "📉 Low-Value Customer",
            4: "⚖️ Average Customer"
        }
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.success(f"**Segment:** {cluster_labels.get(cluster, 'Unknown')}")
        
        with col_b:
            st.metric("Annual Income", f"${income}k")
            st.metric("Spending Score", f"{spending}/100")

# ============================================================================
# PAGE 4: BUSINESS INSIGHTS
# ============================================================================

else:
    st.title("📊 Business Insights & Recommendations")
    st.markdown("---")
    
    # Add cluster labels
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    features_scaled = scaler.transform(features)
    df['Cluster'] = model.predict(features_scaled[:, 1:3])
    
    cluster_names = {
        0: "High-Value Loyal",
        1: "Savers",
        2: "Spending Beyond Means",
        3: "Low-Value",
        4: "Average"
    }
    df['Segment'] = df['Cluster'].map(cluster_names)
    
    st.subheader("Customer Segment Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    segment_counts = df['Segment'].value_counts()
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    bars = ax.bar(segment_counts.index, segment_counts.values, color=colors, edgecolor='black')
    ax.set_title('Customer Segment Distribution', fontsize=16)
    ax.set_xlabel('Segment')
    ax.set_ylabel('Number of Customers')
    
    for bar, count in zip(bars, segment_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(count), ha='center', va='bottom', fontsize=12)
    
    ax.grid(alpha=0.3, axis='y')
    st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Detailed Insights by Segment")
    
    summary_data = df.groupby('Segment').agg({
        'Age': 'mean',
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean',
        'Gender': lambda x: 'Female' if x.value_counts().index[0] == 'Female' else 'Male'
    }).round(2)
    
    summary_data['Count'] = df.groupby('Segment').size()
    summary_data['Percentage'] = (summary_data['Count'] / len(df) * 100).round(1)
    
    st.dataframe(summary_data)
    
    st.markdown("---")
    
    st.subheader("📋 Recommendations")
    
    recommendations = {
        "High-Value Loyal": "🎯 **Priority: HIGHEST**\n\n- Reward with premium loyalty programs\n- Offer exclusive products and early access\n- Provide VIP customer service",
        
        "Savers": "🎯 **Priority: HIGH**\n\n- Showcase value-for-money products\n- Offer bundle deals and discounts\n- Upsell premium items with proven value",
        
        "Spending Beyond Means": "🎯 **Priority: MEDIUM**\n\n- Offer installment payment options\n- Provide discounts on bulk purchases\n- Target with budget-friendly offers",
        
        "Low-Value": "🎯 **Priority: MEDIUM**\n\n- Encourage with introductory offers\n- Send coupons and promotional emails\n- Build engagement with content marketing",
        
        "Average": "🎯 **Priority: STANDARD**\n\n- Maintain satisfaction with good service\n- Occasional upselling opportunities\n- Encourage loyalty through rewards"
    }
    
    for segment, text in recommendations.items():
        with st.expander(f"📌 {segment}"):
            st.markdown(text.replace('\n', '  \n'))

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.caption("Customer Segmentation App")
st.sidebar.caption("Built with Streamlit ❤️")