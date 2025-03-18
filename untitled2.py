import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler

# Streamlit UI Setup
st.set_page_config(page_title="Amazon Analysis", layout="wide")
st.title("Amazon Dataset Interactive Analysis")

# Sidebar for user options
st.sidebar.header("Options")
file_path = "amazon.csv"
df = pd.read_csv(file_path)

# Data Preprocessing
df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)
df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
df['discount_percentage'] = df['discount_percentage'].replace('%', '', regex=True).astype(float)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = df['rating_count'].replace(',', '', regex=True).astype(float)
df.dropna(inplace=True)

# Sidebar - Filtering Options
price_range = st.sidebar.slider("Select Price Range", int(df['discounted_price'].min()),
                                int(df['discounted_price'].max()), (0, 5000))
df_filtered = df[(df['discounted_price'] >= price_range[0]) & (df['discounted_price'] <= price_range[1])]

# Display Dataset Overview
st.subheader("Dataset Overview")
st.dataframe(df_filtered.head(10))

# Visualization Options
viz_option = st.sidebar.selectbox("Choose Analysis", ["Price Distribution", "Price vs Discount", "Correlation Matrix",
                                                      "Customer Segmentation", "Association Rules", "User Behavior"])

if viz_option == "Price Distribution":
    st.subheader("Distribution of Discounted Prices")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(df_filtered['discounted_price'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

elif viz_option == "Price vs Discount":
    st.subheader("Actual Price vs Discounted Price")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(x=df_filtered['actual_price'], y=df_filtered['discounted_price'], ax=ax)
    st.pyplot(fig)

elif viz_option == "Correlation Matrix":
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_filtered[['discounted_price', 'actual_price', 'rating', 'rating_count']].corr(), annot=True,
                cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif viz_option == "Customer Segmentation":
    st.subheader("Customer Segmentation using K-Means")
    features = df_filtered[['discounted_price', 'actual_price', 'rating', 'rating_count']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_filtered['customer_segment'] = kmeans.fit_predict(scaled_features)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df_filtered['discounted_price'], y=df_filtered['actual_price'],
                    hue=df_filtered['customer_segment'], palette='viridis', ax=ax)
    st.pyplot(fig)

elif viz_option == "Association Rules":
    st.subheader("Association Rule Mining")
    basket = df.groupby(['user_id', 'product_name'])['category'].count().unstack().fillna(0)
    basket = basket.map(lambda x: True if x > 0 else False)

    frequent_itemsets = apriori(basket, min_support=0.001, use_colnames=True)
    if frequent_itemsets.empty:
        st.write("No frequent itemsets found. Try lowering min_support.")
    else:
        rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
        st.write("### Top Association Rules")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

elif viz_option == "User Behavior":
    st.subheader("User Behavior Analysis")
    df_filtered['review_content'] = df_filtered['review_content'].astype(str)
    review_lengths = df_filtered['review_content'].apply(len)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(review_lengths, bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# Display Average Rating by Category
st.subheader("Average Rating by Category")
st.write(df.groupby('category')['rating'].mean().sort_values(ascending=False))
