# evaluation.py

import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from utils.preprocessing import load_data, clean_data

# ------------------------------
# Helper functions for evaluation
# ------------------------------

def compute_rmse(y_true, y_pred):
    """Compute the Root Mean Squared Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def precision_at_k(recommended, relevant, k):
    """
    Compute precision@k.
    
    recommended: list of recommended product IDs (ordered)
    relevant: list of relevant (ground truth) product IDs
    k: cutoff rank
    """
    recommended_at_k = recommended[:k]
    if len(recommended_at_k) == 0:
        return 0.0
    intersection = set(recommended_at_k) & set(relevant)
    return len(intersection) / len(recommended_at_k)

def recall_at_k(recommended, relevant, k):
    """
    Compute recall@k.
    
    recommended: list of recommended product IDs (ordered)
    relevant: list of relevant (ground truth) product IDs
    k: cutoff rank
    """
    recommended_at_k = recommended[:k]
    if len(relevant) == 0:
        return 0.0
    intersection = set(recommended_at_k) & set(relevant)
    return len(intersection) / len(relevant)

# ------------------------------
# Loading the pre-trained models
# ------------------------------

def load_models():
    """
    Load all pre-trained models from disk.
    
    Returns:
        cf_model: Collaborative Filtering model (Surprise-based)
        mf_model: Matrix Factorization (SVD) model (Surprise-based)
        content_model: Content-based model
        deep_model: Deep learning model (with its encoders)
    """
    
    with open('models/mf_model.pkl', 'rb') as f:
        mf_model = pickle.load(f)
    
    with open('models/content_model.pkl', 'rb') as f:
        content_model = pickle.load(f)
    
    deep_keras_model = load_model('models/deep_model.h5')
    with open('models/deep_model_encoders.pkl', 'rb') as f:
        deep_encoders = pickle.load(f)
    
    # Create an instance of the DeepRecommender and assign the loaded model and encoders.
    from models.deep_model import DeepRecommender
    deep_model = DeepRecommender()
    deep_model.model = deep_keras_model
    deep_model.user_encoder = deep_encoders['user_encoder']
    deep_model.product_encoder = deep_encoders['product_encoder']
    
    return mf_model, content_model, deep_model

# ------------------------------
# Evaluation for Rating Prediction
# ------------------------------

def evaluate_rating_prediction(model, test_df):
    """
    Evaluate a rating prediction model using RMSE.
    Iterates through test rows and calls model.predict_rating.
    
    Returns the computed RMSE.
    """
    y_true = []
    y_pred = []
    
    for _, row in test_df.iterrows():
        user = row['user_id']
        product = row['product_id']
        # Use the "target" column as ground truth rating.
        true_rating = row['target']
        try:
            pred_rating = model.predict_rating(user, product)
        except Exception as e:
            continue
        y_true.append(true_rating)
        y_pred.append(pred_rating)
        
    rmse = compute_rmse(y_true, y_pred)
    return rmse

# ------------------------------
# Evaluation for Ranking Metrics
# ------------------------------

def evaluate_ranking(model, test_df, k=10, threshold=1):
    """
    Evaluate ranking quality for a given model.
    
    For each user in the test set, we generate a ranked list (using all products from test_df)
    and then compute precision@k and recall@k. Here, we consider a product as relevant if its 
    target value is at least the threshold (for binary/implicit feedback, you might use 1).
    
    Returns:
        avg_precision: Average precision@k across users.
        avg_recall: Average recall@k across users.
    """
    users = test_df['user_id'].unique()
    precision_list = []
    recall_list = []
    
    for user in users:
        # Get ground truth: products the user interacted with (target >= threshold)
        user_data = test_df[test_df['user_id'] == user]
        relevant_products = user_data[user_data['target'] >= threshold]['product_id'].tolist()
        
        # Generate predictions for all products in the test data.
        products = test_df['product_id'].unique()
        preds = []
        for prod in products:
            try:
                pred_rating = model.predict_rating(user, prod)
                preds.append((prod, pred_rating))
            except Exception as e:
                continue
        
        # Sort predicted ratings in descending order.
        preds.sort(key=lambda x: x[1], reverse=True)
        recommended_products = [p[0] for p in preds[:k]]
        
        p_at_k = precision_at_k(recommended_products, relevant_products, k)
        r_at_k = recall_at_k(recommended_products, relevant_products, k)
        
        precision_list.append(p_at_k)
        recall_list.append(r_at_k)
    
    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    return avg_precision, avg_recall

# ------------------------------
# Main evaluation routine
# ------------------------------

def main():
    # Load and clean test data.
    print("Loading test data...")
    _, test_df = load_data('data/train.parquet', 'data/test.parquet')
    test_df = clean_data(test_df)
    print(f"Test data shape: {test_df.shape}")
    
        # Sample the test set to 10K rows (if test_df has more than 10K rows)
    if len(test_df) > 10000:
        test_df = test_df.sample(n=10000, random_state=42)
        print(f"Reduced test data shape: {test_df.shape}")

    # Load all models.
    print("Loading pre-trained models...")
    mf_model, content_model, deep_model = load_models()
    
    print("\n---------------------")
    print("Matrix Factorization (SVD) Model Evaluation:")
    rmse_mf = evaluate_rating_prediction(mf_model, test_df)
    prec_mf, rec_mf = evaluate_ranking(mf_model, test_df, k=10, threshold=1)
    print(f"RMSE: {rmse_mf:.4f}")
    print(f"Precision@10: {prec_mf:.4f}")
    print(f"Recall@10: {rec_mf:.4f}")
    
    print("\n---------------------")
    print("Deep Learning Model Evaluation:")
    rmse_deep = evaluate_rating_prediction(deep_model, test_df)
    prec_deep, rec_deep = evaluate_ranking(deep_model, test_df, k=10, threshold=1)
    print(f"RMSE: {rmse_deep:.4f}")
    print(f"Precision@10: {prec_deep:.4f}")
    print(f"Recall@10: {rec_deep:.4f}")
    
    print("\n---------------------")
    print("Content-Based Model Evaluation (Ranking):")
    # Note: Content-based models don’t provide rating predictions.
    # Here we perform a ranking evaluation using a seed product from each user.
    users = test_df['user_id'].unique()
    precision_list = []
    recall_list = []
    k = 10
    for user in users:
        user_data = test_df[test_df['user_id'] == user]
        if user_data.empty:
            continue
        # Use the first product the user interacted with as a seed.
        seed_product = user_data.iloc[0]['product_id']
        recommended_products = content_model.get_recommendations(seed_product, top_n=k)
        relevant_products = user_data[user_data['target'] >= 1]['product_id'].tolist()
        p_at_k = precision_at_k(recommended_products, relevant_products, k)
        r_at_k = recall_at_k(recommended_products, relevant_products, k)
        precision_list.append(p_at_k)
        recall_list.append(r_at_k)
    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")

if __name__ == '__main__':
    main()
