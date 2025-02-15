# utils/preprocessing.py

import pandas as pd

def load_data(train_path: str, test_path: str):
    """
    Load train and test data from parquet files.
    """
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by handling missing values, duplicates, and data type conversions.
    Also, create a product_description column (for content-based filtering) by concatenating
    the brand and category fields.
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Convert price to numeric (if applicable)
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Fill missing values for categorical columns
    categorical_cols = ['brand', 'cat_0', 'cat_1', 'cat_2', 'cat_3']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Create a product_description column if it doesn't exist
    if 'product_description' not in df.columns:
        df['product_description'] = (
            df['brand'] + " " +
            df['cat_0'] + " " +
            df['cat_1'] + " " +
            df['cat_2'] + " " +
            df['cat_3']
        )
    
    # Forward-fill any remaining missing values
    df = df.fillna(method='ffill')
    
    return df

if __name__ == "__main__":
    train_df, test_df = load_data('data/train.parquet', 'data/test.parquet')
    train_df = clean_data(train_df)
    print("Train data shape:", train_df.shape)
    print("Columns:", train_df.columns.tolist())
