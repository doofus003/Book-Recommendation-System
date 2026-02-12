import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

file_path = "C:/Ines/Sem2/Python/Book_rec/data/data.csv"

def diagnostic_eda():
    try:
        df = pd.read_csv(file_path)
        print(f"\nDataset Shape: {df.shape}\n")

        # Create output folder safely
        output_folder = "eda_outputs"
        os.makedirs(output_folder, exist_ok=True)

        # Drop unnecessary columns for visualization
        df_plot = df.drop(columns=['isbn13', 'isbn10', 'thumbnail'])

        # 1. Missing Data Visualization
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title("Missing Data Map")
        plt.savefig(os.path.join(output_folder, "missing_data.png"))
        plt.show()

        # 2. Rating Distribution
        if 'average_rating' in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df['average_rating'], bins=20, kde=True)
            plt.title("Rating Distribution")
            plt.xlabel("Average Rating")
            plt.ylabel("Count")
            plt.savefig(os.path.join(output_folder, "rating_distribution.png"))
            plt.show()

        # 3. Numerical Feature Analysis
        num_cols = df_plot.select_dtypes(include=['int64', 'float64']).columns

        for col in num_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot: {col}")
            plt.savefig(os.path.join(output_folder, f"boxplot_{col}.png"))
            plt.show()

        # 4. Top 10 Most Rated Books
        if 'ratings_count' in df.columns and 'title' in df.columns:
            top_books = df.sort_values(by='ratings_count', ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='ratings_count', y='title', data=top_books)
            plt.title("Top 10 Most Rated Books")
            plt.savefig(os.path.join(output_folder, "top_10_books.png"))
            plt.show()

        # 5. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        corr = df_plot.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Feature Correlation Heatmap")
        plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
        plt.show()

        # 6. Duplicate Check
        duplicates = df.duplicated().sum()
        print("\nDIAGNOSTIC SUMMARY:")
        print(f"- Total Duplicate Rows: {duplicates}")
        print(f"- Missing Values per Column:\n{df.isnull().sum()}")

        print(f"\nEDA completed successfully. All plots saved in '{output_folder}' folder.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnostic_eda()
