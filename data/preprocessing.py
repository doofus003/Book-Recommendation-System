import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import re

file_path = "C:/Ines/Sem2/Python/Book_rec/data/data.csv"

def preprocess_data():
    try:
        # 1. Load Data
        df = pd.read_csv(file_path)
        print(f"Original Shape: {df.shape}")

        # 2. Remove Exact Duplicates
        df = df.drop_duplicates()
        print(f"After Removing Duplicates: {df.shape}")

        # 3. Drop Unnecessary Columns
        cols_to_drop = ['isbn13', 'isbn10', 'thumbnail']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # 4. Handle Missing Values

        # Drop rows where title is missing
        if 'title' in df.columns:
            df = df.dropna(subset=['title'])

        # Fill numeric missing values with median
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        # Fill categorical missing values with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')

        # 5. Clean Text Columns

        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        text_columns = ['title', 'authors', 'categories', 'description']

        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

        # 6. Remove Unrealistic Values

        if 'average_rating' in df.columns:
            df = df[(df['average_rating'] >= 0) & (df['average_rating'] <= 5)]

        if 'num_pages' in df.columns:
            df = df[df['num_pages'] > 0]

        # 7. Standardize Numerical Features

        scaler = StandardScaler()

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) > 0:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # 8. Reset Index
        df = df.reset_index(drop=True)

        # 9. Save Cleaned Data
        output_folder = "processed_data"
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, "cleaned_data.csv")
        df.to_csv(output_path, index=False)

        print("\nPreprocessing Completed Successfully.")
        print(f"Final Shape: {df.shape}")
        print(f"Cleaned data saved at: {output_path}")

        return df

    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    cleaned_data = preprocess_data()
