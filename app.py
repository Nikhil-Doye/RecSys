# app.py

import streamlit as st
import pandas as pd
from utils.preprocessing import load_data, clean_data
import plotly.express as px

import pickle
from tensorflow.keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_and_clean_data():
    train_df, _ = load_data('data/train.parquet', 'data/test.parquet')
    train_df = clean_data(train_df)
    return train_df

@st.cache(allow_output_mutation=True)
def get_content_based_model():
    with open('models/content_model.pkl', 'rb') as f:
         content_model = pickle.load(f)
    return content_model

@st.cache(allow_output_mutation=True)
def get_matrix_factorization_model():
    with open('models/mf_model.pkl', 'rb') as f:
         mf_model = pickle.load(f)
    return mf_model

@st.cache(allow_output_mutation=True)
def get_deep_model():
    # Load the Keras model and the encoders
    deep_keras_model = load_model('models/deep_model.h5')
    with open('models/deep_model_encoders.pkl', 'rb') as f:
         encoders = pickle.load(f)
    # Create a DeepRecommender instance and assign loaded model and encoders
    from models.deep_model import DeepRecommender
    deep_rec = DeepRecommender()
    deep_rec.model = deep_keras_model
    deep_rec.user_encoder = encoders['user_encoder']
    deep_rec.product_encoder = encoders['product_encoder']
    return deep_rec

def get_recommendations_for_user(model, user_id, df, top_n=5):
    """
    Generate recommendations by predicting ratings for all products.
    """
    products = df['product_id'].unique()
    preds = []
    for product in products:
        try:
            rating = model.predict_rating(user_id, product)
            preds.append((product, rating))
        except Exception as e:
            continue
    preds.sort(key=lambda x: x[1], reverse=True)
    recommendations = [p[0] for p in preds[:top_n]]
    return recommendations

def main():
    st.title("E-commerce Recommendation System")
    st.sidebar.title("Recommendation Settings")
    
    # Load data
    train_df = load_and_clean_data()
    
    model_choice = st.sidebar.selectbox("Select Recommendation Model:", 
                                        ["Content-based", "Matrix Factorization", "Deep Learning"])
    
    if model_choice == "Content-based":
        st.header("Content-Based Recommendation")
        product_id_input = st.text_input("Enter Product ID for Recommendations:")
        if st.button("Get Recommendations"):
            if product_id_input:
                content_model = get_content_based_model()
                recommendations = content_model.get_recommendations(product_id_input)
                if recommendations:
                    st.write("Recommended Products similar to", product_id_input, ":")
                    for rec in recommendations:
                        st.write("- ", rec)
                else:
                    st.write("No recommendations found for the provided Product ID.")
            else:
                st.error("Please enter a valid Product ID.")
    else:
        user_id_input = st.text_input("Enter User ID:")
        if st.button("Get Recommendations"):
            if user_id_input:
                if model_choice == "Matrix Factorization":
                    mf_model = get_matrix_factorization_model()
                    recs = get_recommendations_for_user(mf_model, user_id_input, train_df)
                elif model_choice == "Deep Learning":
                    deep_model = get_deep_model()
                    recs = get_recommendations_for_user(deep_model, user_id_input, train_df)
                if recs:
                    st.write("Top Recommendations for User", user_id_input, ":")
                    for rec in recs:
                        st.write("- ", rec)
                else:
                    st.write("No recommendations found for the provided User ID.")
            else:
                st.error("Please enter a valid User ID.")
    
    # Display an interactive visualization of product popularity
    st.header("Product Popularity")
    product_counts = train_df['product_id'].value_counts().reset_index()
    product_counts.columns = ['product_id', 'interactions']
    fig = px.bar(product_counts.head(20), x='product_id', y='interactions', title='Top 20 Popular Products')
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
