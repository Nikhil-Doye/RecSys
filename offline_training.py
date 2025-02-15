# offline_training.py

import os
import pickle
from utils.preprocessing import load_data, clean_data
from models.content_based import ContentBasedRecommender
#from models.collaborative import CollaborativeFiltering
from models.matrix_factorization import MatrixFactorization
from models.deep_model import DeepRecommender

def main():
    # Load and clean training data
    print("Loading and cleaning data...")
    train_df, _ = load_data('data/train.parquet', 'data/test.parquet')
    train_df = clean_data(train_df)
    
    # --- Train and Save Content-Based Model ---
    print("Training Content-Based Model...")
    # Use product_id and the new product_description column
    products_df = train_df[['product_id', 'product_description']].drop_duplicates().reset_index(drop=True)
    content_model = ContentBasedRecommender(products_df, n_components=100, n_trees=10)
    with open('models/content_model.pkl', 'wb') as f:
        pickle.dump(content_model, f)
    
    # --- Train and Save Matrix Factorization (SVD) Model ---
    print("Training Matrix Factorization (SVD) Model...")
    mf_model = MatrixFactorization()
    mf_model.prepare_data(train_df, user_col='user_id', item_col='product_id', rating_col='target')
    mf_model.train()
    with open('models/mf_model.pkl', 'wb') as f:
        pickle.dump(mf_model, f)
    
    # --- Train and Save Deep Learning Model ---
    print("Training Deep Learning Model...")
    deep_model = DeepRecommender(embedding_size=20, epochs=5, batch_size=256)  # Adjust epochs as needed
    # Again, use "target" for ratings
    X, y = deep_model.prepare_data(train_df, user_col='user_id', item_col='product_id', rating_col='target')
    deep_model.train(X, y)
    # Save the Keras model
    deep_model.model.save('models/deep_model.h5')
    # Save the encoders (needed for processing new inputs)
    with open('models/deep_model_encoders.pkl', 'wb') as f:
        pickle.dump({
            'user_encoder': deep_model.user_encoder,
            'product_encoder': deep_model.product_encoder
        }, f)
    
    print("Offline training completed. Models saved in the 'models/' directory.")

if __name__ == '__main__':
    main()
