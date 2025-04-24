import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# Load the dataset
data = pd.read_csv(r'D:\summa\IRIS.csv')  
print(data.head())
# Select only numeric columns and handle missing values
data = data.select_dtypes(include='number')
data = data.fillna(data.mean())  
# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)
# Error Measures
kmeans_inertia = kmeans.inertia_
kmeans_sil_score = silhouette_score(scaled_data, kmeans_labels)
# Print the error measures
print(f"K-Means Inertia: {kmeans_inertia}")
print(f"K-Means Silhouette Score: {kmeans_sil_score}")