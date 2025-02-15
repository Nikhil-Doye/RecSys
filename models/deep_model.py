# models/deep_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DeepRecommender:
    def __init__(self, embedding_size=20, epochs=5, batch_size=256):
        """
        Initialize the deep recommender with a simplified architecture.
        The model uses a dot product of user and product embeddings to predict ratings.
        
        Parameters:
            embedding_size (int): Dimension of the embedding vectors.
            epochs (int): Maximum number of training epochs.
            batch_size (int): Batch size for training.
        """
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.num_users = None
        self.num_products = None

    def prepare_data(self, df: pd.DataFrame, user_col='user_id', item_col='product_id', rating_col='rating'):
        """
        Encode user and product IDs as sequential integers and prepare training arrays.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            user_col (str): Column name for user IDs.
            item_col (str): Column name for product IDs.
            rating_col (str): Column name for target ratings (or implicit feedback).
        
        Returns:
            X (np.array): Array of shape (n_samples, 2) with encoded user and product IDs.
            y (np.array): Array of target ratings.
        """
        # Encode user and product IDs to integers
        df[user_col] = self.user_encoder.fit_transform(df[user_col])
        df[item_col] = self.product_encoder.fit_transform(df[item_col])
        self.num_users = df[user_col].nunique()
        self.num_products = df[item_col].nunique()

        X = df[[user_col, item_col]].values
        y = df[rating_col].values
        return X, y

    def build_model(self):
        """
        Build and compile the simplified deep learning model.
        The model uses embeddings for users and products, followed by a dot product.
        """
        # Define input layers for user and product IDs.
        user_input = Input(shape=(1,), name='user_input')
        product_input = Input(shape=(1,), name='product_input')

        # Create embedding layers.
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_size,
                                   name='user_embedding')(user_input)
        product_embedding = Embedding(input_dim=self.num_products, output_dim=self.embedding_size,
                                      name='product_embedding')(product_input)

        # Flatten the embeddings.
        user_vec = Flatten()(user_embedding)
        product_vec = Flatten()(product_embedding)

        # Use dot product to capture interaction.
        dot_product = Dot(axes=1, name='dot_product')([user_vec, product_vec])

        # Build and compile the model.
        self.model = Model(inputs=[user_input, product_input], outputs=dot_product)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        print(self.model.summary())

    def train(self, X, y):
        """
        Build and train the model on the provided data.
        Uses early stopping to avoid unnecessary epochs.
        
        Parameters:
            X (np.array): Input array with encoded user and product IDs.
            y (np.array): Target ratings.
        """
        self.build_model()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        self.model.fit([X_train[:, 0], X_train[:, 1]], y_train,
                       validation_data=([X_val[:, 0], X_val[:, 1]], y_val),
                       epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=[early_stop])
    
    def predict_rating(self, user_id, product_id):
        """
        Predict the rating for a given user and product.
        
        Parameters:
            user_id: The raw user ID (before encoding).
            product_id: The raw product ID (before encoding).
        
        Returns:
            A predicted rating (float).
        """
        # Encode the inputs using the previously fitted encoders.
        user_encoded = self.user_encoder.transform([user_id])
        product_encoded = self.product_encoder.transform([product_id])
        pred = self.model.predict([user_encoded, product_encoded])
        return pred[0][0]

if __name__ == "__main__":
    # Load your dataset. For an e-commerce dataset, you might use 'target' as the rating column.
    df = pd.read_parquet('data/train.parquet')
    
    # Create an instance of the recommender.
    # If your dataset uses "target" for ratings, update the parameter accordingly.
    deep_rec = DeepRecommender(embedding_size=20, epochs=3, batch_size=256)
    
    # Prepare the data; if your rating column is named 'target', change rating_col below.
    X, y = deep_rec.prepare_data(df, user_col='user_id', item_col='product_id', rating_col='target')
    
    # Train the model.
    deep_rec.train(X, y)
    
    # Get a sample prediction.
    user_example = df['user_id'].iloc[0]
    product_example = df['product_id'].iloc[0]
    print(f"Deep Model predicted rating for user {user_example} and product {product_example}: {deep_rec.predict_rating(user_example, product_example)}")