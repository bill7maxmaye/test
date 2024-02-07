import pandas as pd
from sklearn.cluster import KMeans

# Load the customer dataset from CSV
data = pd.read_csv('customer_data.csv')

# Preprocess the data if needed (e.g., handle missing values, scale features)

# Select the relevant features for clustering
features = data[['feature1', 'feature2', 'feature3']]

# Train the k-means clustering model
kmeans = KMeans(n_clusters=3)  # Specify the number of clusters
kmeans.fit(features)

# Get the cluster labels for each data point
cluster_labels = kmeans.labels_

# Add the cluster labels to the original dataset
data['cluster'] = cluster_labels

# Save the updated dataset with cluster labels to a new CSV file
data.to_csv('customer_data_with_clusters.csv', index=False)