# utils/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_user_purchase_distribution(df: pd.DataFrame, user_column='user_id'):
    """
    Plot the distribution of user interactions.
    """
    user_counts = df[user_column].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_counts, bins=50, kde=True)
    plt.title('Distribution of User Interactions')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users')
    plt.show()

def plot_product_popularity(df: pd.DataFrame, product_column='product_id'):
    """
    Plot the product popularity distribution.
    """
    product_counts = df[product_column].value_counts()
    fig = px.bar(x=product_counts.index, y=product_counts.values,
                 labels={'x': 'Product ID', 'y': 'Interactions'},
                 title='Product Popularity')
    fig.show()

if __name__ == "__main__":
    # For testing purposes, load some data and run EDA functions
    df = pd.read_parquet('data/train.parquet')
    plot_user_purchase_distribution(df)
    plot_product_popularity(df)
