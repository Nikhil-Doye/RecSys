# E-commerce Recommendation System with Offline Training

This project implements a personalized recommendation system for an e-commerce platform. It includes data preprocessing, exploratory data analysis (EDA), several recommendation models (content-based, collaborative filtering, matrix factorization, and deep learning), and an interactive UI built with Streamlit.

### Project Structure

project/
├── data/
│ ├── train.parquet
│ └── test.parquet
├── models/
│ ├── content_based.py
│ ├── collaborative.py
│ ├── matrix_factorization.py
│ └── deep_model.py
├── utils/
│ ├── eda.py
│ └── preprocessing.py
├── offline_training.py
├── app.py
├── requirements.txt
└── README.md

### Setup and Installation

1. **Clone the repository.**

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt

   ```

3. **Offline Training:**

   ```bash
   python offline_training.py

   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```
