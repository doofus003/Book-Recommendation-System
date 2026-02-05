import pandas as pd
import numpy as np

file_path = "C:/Ines/Sem2/Python/Book_rec/data/data.csv"

print(" STARTING DATA CLEANING PROCESS\n")

#### Loading data ####
try:
    data = pd.read_csv(file_path)
    print(" Data loaded successfully!")
    print(f" Shape: {data.shape}")
    print(f" Columns: {list(data.columns)}")

except FileNotFoundError:
    print(f" File not found: {file_path}")
    data = None

except Exception as e:
    print(f" Error loading file: {e}")
    data = None

#### Data Cleaning ####

if data is not None:

    print("\n ORIGINAL DATA INFO:")
    print(data.info())

    print("\n MISSING VALUES PER COLUMN:")
    print(data.isnull().sum())

    # Drop rows with any missing values
    data_cleaned = data.dropna()

    print("\n AFTER DROPPING NULL VALUES:")
    print(f" Rows before: {len(data)}")
    print(f" Rows after:  {len(data_cleaned)}")
    print(f" Dropped rows: {len(data) - len(data_cleaned)}")

    # Save cleaned data
    output_file = "C:/Ines/Sem2/Python/Book_rec/data/data_cleaned.csv"
    data_cleaned.to_csv(output_file, index=False)

    print(f" Final shape: {data_cleaned.shape}")
    print("\n DATA CLEANING COMPLETED SUCCESSFULLY!")

else:
    print("\n DATA CLEANING FAILED - No data loaded")
