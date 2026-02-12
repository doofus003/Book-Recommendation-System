import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = "C:/Ines/Sem2/Python/Book_rec/data/data.csv"

def diagnostic_eda():
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset Shape: {df.shape}\n")

        # 1. MISSING DATA VISUALIZATION
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title("Missing Data Map (Yellow lines = Missing)")
        plt.show()
        
        # Column is mostly yellow = bad, drop it whole instead of rows

        # 2. NUMERICAL OUTLIERS & SKEWNESS
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(8,6))
            
            sns.boxplot(x=df[col], ax=ax_box, color="salmon")
            sns.histplot(df[col], ax=ax_hist, kde=True)
            
            ax_box.set(title=f"Diagnostic: {col}")
            plt.show()
            
            # Interpretation
            # Boxplot: Dots outside the 'whiskers' are outliers; are they valid or errors
            # Histogram: If the 'tail' is very long, you may need to apply a Log Transformation

        # 3. CATEGORICAL INCONSISTENCY (Data Entry Errors)
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].nunique() < 30:  # Focus on columns with limited categories
                plt.figure(figsize=(10, 5))
                df[col].value_counts().plot(kind='barh', color='teal')
                plt.title(f"Category Counts: {col}")
                plt.xlabel("Frequency")
                plt.show()
                # Interpretation
                # Look for duplicate categories with different spelling (e.g., 'Fiction' vs 'fiction').")
                # Look for placeholder values like 'None', 'Unknown', or '?'.\n")

        # 4. DUPLICATE CHECK
        duplicates = df.duplicated().sum()
        print(f"DIAGNOSTIC SUMMARY:")
        print(f"- Total Duplicate Rows: {duplicates}")
        print(f"- Missing Values Count:\n{df.isnull().sum()}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnostic_eda()