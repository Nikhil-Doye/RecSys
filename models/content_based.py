# models/content_based_ann.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from annoy import AnnoyIndex
import numpy as np

class ContentBasedRecommender:
    def __init__(self, products_df: pd.DataFrame, text_column='product_description', n_components=100, n_trees=10):
        """
        Initialize the recommender that uses an ANN approach.
        
        Steps:
        1. Compute the TF-IDF matrix from the product descriptions.
        2. Reduce the dimensionality of the TF-IDF matrix using TruncatedSVD.
        3. Build an Annoy index on the reduced vectors.
        
        Parameters:
            products_df (pd.DataFrame): DataFrame containing product information.
            text_column (str): Column name containing the text to vectorize.
            n_components (int): Number of components for TruncatedSVD.
            n_trees (int): Number of trees for building the Annoy index.
        """
        self.products_df = products_df.reset_index(drop=True)
        self.text_column = text_column
        self.n_components = n_components
        self.n_trees = n_trees

        # Step 1: Compute TF-IDF vectors.
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.products_df[self.text_column])

        # Step 2: Reduce dimensionality to make nearest neighbor search more efficient.
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.reduced_matrix = self.svd.fit_transform(self.tfidf_matrix)

        # Step 3: Build an Annoy index for the reduced vectors.
        self._build_annoy_index()

    def _build_annoy_index(self):
        self.index = AnnoyIndex(self.n_components, metric='angular')
        for i, vec in enumerate(self.reduced_matrix):
            self.index.add_item(i, vec.tolist())
        self.index.build(self.n_trees)

    def get_recommendations(self, product_id, top_n=10):
        """
        Given a product_id, return the top_n most similar products using the Annoy index.
        
        Parameters:
            product_id (str): The product identifier for which to find similar items.
            top_n (int): Number of similar products to return.
            
        Returns:
            List[str]: List of recommended product IDs.
        """
        # Find the index (row number) of the product.
        indices = self.products_df.index[self.products_df['product_id'] == product_id].tolist()
        if not indices:
            return []
        idx = indices[0]
        
        # Query the Annoy index for nearest neighbors.
        neighbor_indices = self.index.get_nns_by_item(idx, top_n + 1, include_distances=False)
        # Exclude the product itself (if present).
        neighbor_indices = [i for i in neighbor_indices if i != idx][:top_n]
        
        recommended_products = self.products_df.iloc[neighbor_indices]['product_id'].tolist()
        return recommended_products

    def __getstate__(self):
        """Customize pickling: exclude the Annoy index."""
        state = self.__dict__.copy()
        # Remove the Annoy index from the state (it is not pickleable)
        if 'index' in state:
            del state['index']
        return state

    def __setstate__(self, state):
        """Rebuild the Annoy index after unpickling."""
        self.__dict__.update(state)
        # Rebuild the Annoy index
        self._build_annoy_index()


# Example usage:
if __name__ == '__main__':
    # For testing purposes, load your dataset.
    df = pd.read_parquet('data/train.parquet')
    # Ensure that your DataFrame contains 'product_id' and 'product_description'.
    # (If not, use your preprocessing step to create 'product_description'.)
    products_df = df[['product_id', 'product_description']].drop_duplicates().reset_index(drop=True)
    recommender = ContentBasedRecommender(products_df, n_components=100, n_trees=10)
    
    # Get recommendations for the first product.
    sample_product_id = products_df.iloc[0]['product_id']
    recommendations = recommender.get_recommendations(sample_product_id)
    print("Recommendations for product", sample_product_id, ":", recommendations)
