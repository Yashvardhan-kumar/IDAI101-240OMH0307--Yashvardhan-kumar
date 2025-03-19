import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
import os
import sys
import pyarrow as pa
import warnings

# Suppress specific warnings related to missing ScriptRunContext
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# Suppress matplotlib "More than 20 figures have been opened" warning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="More than 20 figures have been opened.*")

# Ensure the virtual environment is activated
venv_path = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
if sys.executable != venv_path:
    st.error("⚠️ Please activate the virtual environment before running the script.")
    st.stop()

# Streamlit Page Config
st.set_page_config(page_title="Amazon Data Analysis", layout="wide")

# Sidebar Navigation
st.sidebar.header("Amazon Dataset Analysis")
st.sidebar.write("🔍 Select an analysis to explore the dataset.")

# Load dataset with caching
@st.cache_data
def load_data():
    try:
        file_path = "/content/amazon.csv"
        df = pd.read_csv(file_path)

        # Data Cleaning
        df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)
        df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
        df['discount_percentage'] = df['discount_percentage'].replace('%', '', regex=True).astype(float)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating_count'] = df['rating_count'].replace(',', '', regex=True).astype(float)

        # Handle missing values
        df.dropna(inplace=True)

        # Ensure correct data types (e.g., string for categorical columns)
        df['category'] = df['category'].astype(str)

        return df
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        return None

df = load_data()

if df is not None:
    # Dataset Overview
    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    # Exploratory Data Analysis (EDA)
    st.subheader("🔬 Exploratory Data Analysis")

    analysis_option = st.selectbox(
        "Select Analysis Type:", 
        ["Discounted Price Distribution", "Price Comparison", "Correlation Matrix"]
    )

    if analysis_option == "Discounted Price Distribution":
        st.write("### Distribution of Discounted Prices")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.histplot(df['discounted_price'], bins=50, kde=True, ax=ax)
        st.pyplot(fig)
        #plt.close(fig)  # Close the figure to avoid memory issue

    elif analysis_option == "Price Comparison":
        st.write("### Actual Price vs Discounted Price")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.scatterplot(x=df['actual_price'], y=df['discounted_price'], ax=ax)
        st.pyplot(fig)
        #plt.close(fig)  # Close the figure to avoid memory issue

    elif analysis_option == "Correlation Matrix":
        st.write("### Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[['discounted_price', 'actual_price', 'rating', 'rating_count']].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        #plt.close(fig)  # Close the figure to avoid memory issue

    # Customer Segmentation using K-Means
    st.subheader("🎯 Customer Segmentation")

    try:
        features = df[['discounted_price', 'actual_price', 'rating', 'rating_count']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=3, random_state=42)
        df['customer_segment'] = kmeans.fit_predict(scaled_features)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df['discounted_price'], y=df['actual_price'], hue=df['customer_segment'], palette='viridis', ax=ax)
        st.pyplot(fig)
        #plt.close(fig)  # Close the figure to avoid memory issue
    except Exception as e:
        st.error(f"⚠️ Error in K-Means Clustering: {e}")

    # Association Rule Mining
    st.subheader("📈 Association Rule Mining")
    try:
        basket = df.groupby(['user_id', 'product_name'])['category'].count().unstack().fillna(0)
        basket = basket.map(lambda x: True if x > 0 else False)

        frequent_itemsets = apriori(basket, min_support=0.001, use_colnames=True)
        if frequent_itemsets.empty:
            st.write("⚠️ No frequent itemsets found. Try lowering min_support.")
        else:
            rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
            
            # Convert frozenset to string for compatibility with Arrow
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            st.write("### Top Association Rules")
            st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    except Exception as e:
        st.error(f"⚠️ Error in Association Rule Mining: {e}")

    # User Behavior Analysis
    st.subheader("📊 User Behavior Analysis")
    try:
        df['review_content'] = df['review_content'].fillna("")
        review_lengths = df['review_content'].apply(lambda x: len(str(x)))

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(review_lengths, bins=30, kde=True, ax=ax)
        st.pyplot(fig)
        #plt.close(fig)  # Close the figure to avoid memory issue
    except Exception as e:
        st.error(f"⚠️ Error in User Behavior Analysis: {e}")

    # Average Rating by Category
    st.subheader("⭐ Average Rating by Category")
    try:
        if "category" in df.columns:
            avg_ratings = df.groupby('category')['rating'].mean().sort_values(ascending=False)
            st.write(avg_ratings)
        else:
            st.write("⚠️ Category column missing in dataset.")
    except Exception as e:
        st.error(f"⚠️ Error in Average Rating Calculation: {e}")

    # Check Arrow Conversion
    try:
        table = pa.Table.from_pandas(df)
        st.write("Arrow conversion successful.")
    except Exception as e:
        st.error(f"⚠️ Arrow conversion error: {e}")
