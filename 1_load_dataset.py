# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (Change 'dataset.csv' to your actual file)
df = pd.read_csv("dataset.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:\n", df.head())

# Dataset information (data types, missing values, etc.)
print("\nDataset Info:")
print(df.info())

# Summary statistics (Numerical columns)
print("\nSummary Statistics:")
print(df.describe())

# Checking for missing values
print("\nMissing values in each column:\n", df.isnull().sum())

# Correlation heatmap (for numerical features)
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
