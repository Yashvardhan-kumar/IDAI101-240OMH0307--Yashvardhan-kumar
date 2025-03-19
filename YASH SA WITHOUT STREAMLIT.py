import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress specific warnings related to missing ScriptRunContext
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# Suppress matplotlib "More than 20 figures have been opened" warning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="More than 20 figures have been opened.*")

# Load dataset
def load_data():
    try:
        file_path = "C:\\Users\\admin\\Downloads\\amazon.csv"  # Change path if the file is different in your setup
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
        print(f"Error loading dataset: {e}")
        return None

df = load_data()

# Dataset Overview
if df is not None:
    # Display the first few rows of the dataset
    print("Dataset Overview:")
    print(df.head())

    # Exploratory Data Analysis (EDA)
    print("\nExploratory Data Analysis:")

    analysis_option = "Discounted Price Distribution"  # You can change analysis options as needed

    if analysis_option == "Discounted Price Distribution":
        print("### Distribution of Discounted Prices")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.histplot(df['discounted_price'], bins=50, kde=True, ax=ax)
        plt.show()  # Display the plot in Google Colab

    elif analysis_option == "Price Comparison":
        print("### Actual Price vs Discounted Price")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.scatterplot(x=df['actual_price'], y=df['discounted_price'], ax=ax)
        plt.show()  # Display the plot in Google Colab

    elif analysis_option == "Correlation Matrix":
        print("### Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[['discounted_price', 'actual_price', 'rating', 'rating_count']].corr(), annot=True, cmap='coolwarm', ax=ax)
        plt.show()  # Display the plot in Google Colab

    # Customer Segmentation using K-Means
    print("\nCustomer Segmentation using K-Means")

    try:
        features = df[['discounted_price', 'actual_price', 'rating', 'rating_count']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=3, random_state=42)
        df['customer_segment'] = kmeans.fit_predict(scaled_features)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df['discounted_price'], y=df['actual_price'], hue=df['customer_segment'], palette='viridis', ax=ax)
        plt.show()  # Display the plot in Google Colab
    except Exception as e:
        print(f"Error in K-Means Clustering: {e}")

    # Association Rule Mining
    print("\nAssociation Rule Mining")
    try:
        basket = df.groupby(['user_id', 'product_name'])['category'].count().unstack().fillna(0)
        basket = basket.map(lambda x: True if x > 0 else False)

        frequent_itemsets = apriori(basket, min_support=0.001, use_colnames=True)
        if frequent_itemsets.empty:
            print("No frequent itemsets found. Try lowering min_support.")
        else:
            rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
            
            # Convert frozenset to string for compatibility with Arrow
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            print("### Top Association Rules")
            print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    except Exception as e:
        print(f"Error in Association Rule Mining: {e}")

    # User Behavior Analysis
    print("\nUser Behavior Analysis")
    try:
        df['review_content'] = df['review_content'].fillna("")
        review_lengths = df['review_content'].apply(lambda x: len(str(x)))

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(review_lengths, bins=30, kde=True, ax=ax)
        plt.show()  # Display the plot in Google Colab
    except Exception as e:
        print(f"Error in User Behavior Analysis: {e}")

    # Average Rating by Category
    print("\nAverage Rating by Category")
    try:
        if "category" in df.columns:
            avg_ratings = df.groupby('category')['rating'].mean().sort_values(ascending=False)
            print(avg_ratings)
        else:
            print("Category column missing in dataset.")
    except Exception as e:
        print(f"Error in Average Rating Calculation: {e}")

    # Check Arrow Conversion
    try:
        import pyarrow as pa
        table = pa.Table.from_pandas(df)
        print("Arrow conversion successful.")
    except Exception as e:
        print(f"Error in Arrow conversion: {e}")
