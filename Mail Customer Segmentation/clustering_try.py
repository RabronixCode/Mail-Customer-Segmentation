from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

df = pd.read_csv('Mall_Customers.csv')

# Select the features for clustering
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Define a function to evaluate clustering models
def evaluate_clustering(labels, data, method_name):
    score = silhouette_score(data, labels)
    print(f"Silhouette Score for {method_name}: {score:.2f}")
    return score

# Pipeline for K-means
kmeans_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=6, random_state=42))
])

# Fit and predict K-means
kmeans_pipeline.fit(features)
df['KMeans_Cluster'] = kmeans_pipeline['kmeans'].labels_

# Evaluate K-means
evaluate_clustering(df['KMeans_Cluster'], kmeans_pipeline['scaler'].transform(features), 'K-means')

# Pipeline for DBSCAN
dbscan_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('dbscan', DBSCAN(eps=0.5, min_samples=5))
])

# Fit and predict DBSCAN
dbscan_pipeline.fit(features)
df['DBSCAN_Cluster'] = dbscan_pipeline['dbscan'].labels_

# Evaluate DBSCAN (Note: DBSCAN might assign some points to noise, labeled as -1)
dbscan_labels = df['DBSCAN_Cluster']
if len(set(dbscan_labels)) > 1:  # Only evaluate if more than one cluster is formed
    evaluate_clustering(dbscan_labels, dbscan_pipeline['scaler'].transform(features), 'DBSCAN')
else:
    print("DBSCAN did not find enough clusters to evaluate.")

# Pipeline for Gaussian Mixture Model (GMM)
gmm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gmm', GaussianMixture(n_components=6, random_state=42))
])

# Fit and predict GMM
gmm_pipeline.fit(features)
df['GMM_Cluster'] = gmm_pipeline['gmm'].predict(gmm_pipeline['scaler'].transform(features))

# Evaluate GMM
evaluate_clustering(df['GMM_Cluster'], gmm_pipeline['scaler'].transform(features), 'GMM')

# Optional: Visualize the clustering results (e.g., PCA plot or scatter plot)
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the clusters for K-means
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='KMeans_Cluster', data=df, palette='tab10')
plt.title('K-means Clustering')
plt.show()

# Visualize the clusters for DBSCAN
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='DBSCAN_Cluster', data=df, palette='tab10')
plt.title('DBSCAN Clustering')
plt.show()

# Visualize the clusters for GMM
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='GMM_Cluster', data=df, palette='tab10')
plt.title('Gaussian Mixture Model (GMM) Clustering')
plt.show()


