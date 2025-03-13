# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load and Explore the Dataset
df = pd.read_csv("dataset.csv")  # Change this to your actual dataset

print("First 5 rows of the dataset:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Checking for missing values
print("\nMissing values:\n", df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 2: Data Preprocessing
# Handling missing values
df.fillna(df.mean(), inplace=True)  # Numerical columns
df.fillna(df.mode().iloc[0], inplace=True)  # Categorical columns

# Encoding categorical variables (if needed)
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Splitting into features and target variable
X = df.drop("target", axis=1)  # Replace 'target' with your actual target column name
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Model Selection
# Choose a model based on the problem type
if y.nunique() > 10:  # Assuming continuous target means regression
    print("\nApplying Regression Model")
    model = LinearRegression()
else:
    print("\nApplying Classification Model")
    model = LogisticRegression()

# Training the model
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)

if y.nunique() > 10:  # Regression Evaluation
    print("\nRegression Model Evaluation:")
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))
else:  # Classification Evaluation
    print("\nClassification Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 5: Inference
print("\nInference:")
if y.nunique() > 10:
    print("The regression model predicts the target variable with an R² score of", round(r2_score(y_test, y_pred), 2))
else:
    print("The classification model achieved an accuracy of", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
