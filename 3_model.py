from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Select model type
if y.nunique() > 10:  # If target is numerical, use Regression
    print("\nApplying Regression Model")
    model = LinearRegression()
else:  # If target is categorical, use Classification
    print("\nApplying Classification Model")
    model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel Training Completed ✅")
