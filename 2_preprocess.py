from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Handle missing values
df.fillna(df.mean(), inplace=True)  # Fill missing values in numerical columns
df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical columns

# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define Features (X) and Target (y)
X = df.drop("target", axis=1)  # Change 'target' to your actual target column name
y = df["target"]

# Split into training and testing datasets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (for better model performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData Preprocessing Completed ✅")
