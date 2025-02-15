# models/matrix_factorization.py

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

class MatrixFactorization:
    def __init__(self, rating_scale=(1, 5)):
        self.rating_scale = rating_scale
        self.algo = SVD()
        self.trainset = None
        self.testset = None

    def prepare_data(self, df: pd.DataFrame, user_col='user_id', item_col='product_id', rating_col='rating'):
        """
        Prepare the data for matrix factorization.
        """
        from surprise import Dataset, Reader
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(df[[user_col, item_col, rating_col]], reader)
        self.trainset, self.testset = train_test_split(data, test_size=0.25)
        return self.trainset, self.testset

    def train(self):
        """
        Train the SVD model.
        """
        self.algo.fit(self.trainset)

    def evaluate(self):
        """
        Evaluate the model performance using RMSE.
        """
        predictions = self.algo.test(self.testset)
        rmse = accuracy.rmse(predictions, verbose=True)
        return rmse

    def predict_rating(self, user_id, product_id):
        """
        Predict the rating for a given user and product.
        """
        prediction = self.algo.predict(user_id, product_id)
        return prediction.est

if __name__ == "__main__":
    df = pd.read_parquet('data/train.parquet')
    mf = MatrixFactorization()
    mf.prepare_data(df)
    mf.train()
    print("Matrix Factorization (SVD) Model RMSE:", mf.evaluate())
    user_example = df['user_id'].iloc[0]
    product_example = df['product_id'].iloc[0]
    print(f"Predicted rating for user {user_example} and product {product_example}: {mf.predict_rating(user_example, product_example)}")
