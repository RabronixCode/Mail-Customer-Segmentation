from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

df = pd.read_csv('Mall_Customers.csv')

print(df.isnull().sum())
print(df.info())
print(df.describe())

# Set up the plotting environment
plt.figure(figsize=(18, 5))

# Histogram for Age
plt.subplot(1, 3, 1)
plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Histogram for Annual Income
plt.subplot(1, 3, 2)
plt.hist(df['Annual Income (k$)'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')

# Histogram for Spending Score
plt.subplot(1, 3, 3)
plt.hist(df['Spending Score (1-100)'], bins=20, color='salmon', edgecolor='black')
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Set up the plotting environment
plt.figure(figsize=(18, 5))

# Boxplot for Age
plt.subplot(1, 3, 1)
plt.boxplot(df['Age'], patch_artist=True)
plt.title('Age Distribution')
plt.ylabel('Age')

# Boxplot for Annual Income
plt.subplot(1, 3, 2)
plt.boxplot(df['Annual Income (k$)'], patch_artist=True)
plt.title('Annual Income Distribution')
plt.ylabel('Annual Income (k$)')

# Boxplot for Spending Score
plt.subplot(1, 3, 3)
plt.boxplot(df['Spending Score (1-100)'], patch_artist=True)
plt.title('Spending Score Distribution')
plt.ylabel('Spending Score (1-100)')

plt.show()

#______________________________________

# Select the features for clustering
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert scaled features back to a DataFrame for easier handling
scaled_df = pd.DataFrame(scaled_features, columns=['Age (scaled)', 'Annual Income (scaled)', 'Spending Score (scaled)'])

print(scaled_df.head())


# Determine the optimal number of clusters using the Elbow Method
inertia = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Perform Silhouette Analysis
silhouette_scores = []

for k in cluster_range[1:]:  # Start from 2 clusters because silhouette score is not defined for 1 cluster
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_scores.append(score)

# Plot Silhouette Scores
plt.figure(figsize=(8, 6))
plt.plot(cluster_range[1:], silhouette_scores, marker='o')
plt.title('Silhouette Analysis For Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

db_scores = []
ch_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    
    labels = kmeans.labels_
    
    # Calculate Davies-Bouldin Index
    db_score = davies_bouldin_score(scaled_features, labels)
    db_scores.append(db_score)
    
    # Calculate Calinski-Harabasz Index
    ch_score = calinski_harabasz_score(scaled_features, labels)
    ch_scores.append(ch_score)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 11), db_scores, marker='o')
plt.title('Davies-Bouldin Index by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Index')
plt.gca().invert_yaxis()  # Lower values are better

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), ch_scores, marker='o')
plt.title('Calinski-Harabasz Index by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Index')

plt.tight_layout()
plt.show()


# Apply K-means clustering with 6 clusters
kmeans_6 = KMeans(n_clusters=6, init="k-means++", n_init=100, max_iter=500, random_state=42)
df['Cluster'] = kmeans_6.fit_predict(scaled_features)

# View the first few rows with cluster assignments
print(df.head())

# Visualize the clusters using a scatter plot of Annual Income vs Spending Score
plt.figure(figsize=(10, 8))

# Plot each cluster with a different color
for cluster in range(6):
    plt.scatter(
        df[df['Cluster'] == cluster]['Annual Income (k$)'],
        df[df['Cluster'] == cluster]['Spending Score (1-100)'],
        label=f'Cluster {cluster}'
    )

# Plot the centroids
centroids = kmeans_6.cluster_centers_
plt.scatter(
    centroids[:, 1] * scaler.scale_[1] + scaler.mean_[1],  # Annual Income
    centroids[:, 2] * scaler.scale_[2] + scaler.mean_[2],  # Spending Score
    s=300, c='black', marker='x', label='Centroids'
)

plt.title('Customer Segmentation Using K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

#_______________________________

# Prepare data for radar chart
cluster_centers = scaler.inverse_transform(kmeans_6.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

# Number of variables we're plotting
categories = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Initialize the radar plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
plt.ylim(0, 100)

# Plot each cluster
for i in range(len(cluster_df)):
    values = cluster_df.iloc[i].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {i}')
    ax.fill(angles, values, alpha=0.25)

# Add a legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.title('Cluster Comparison Across All Features')
plt.show()

#___________________________________

# Set up the plotting environment
plt.figure(figsize=(18, 12))

# Boxplot for Age by Cluster
plt.subplot(3, 1, 1)
sns.boxplot(x='Cluster', y='Age', data=df)
plt.title('Age Distribution per Cluster')

# Boxplot for Annual Income by Cluster
plt.subplot(3, 1, 2)
sns.boxplot(x='Cluster', y='Annual Income (k$)', data=df)
plt.title('Annual Income Distribution per Cluster')

# Boxplot for Spending Score by Cluster
plt.subplot(3, 1, 3)
sns.boxplot(x='Cluster', y='Spending Score (1-100)', data=df)
plt.title('Spending Score Distribution per Cluster')

plt.tight_layout()
plt.show()

#_____________________________________

# Inverse transform the scaled centroids to interpret them in the original feature space
cluster_centers = scaler.inverse_transform(kmeans_6.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

# Group the original DataFrame by cluster and calculate descriptive statistics
cluster_summary = df.groupby('Cluster').agg({
    'Age': ['mean', 'std', 'min', 'max'],
    'Annual Income (k$)': ['mean', 'std', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'std', 'min', 'max']
}).reset_index()

# Display the cluster summary
print("Cluster Summary Statistics:")
print(cluster_summary)

#___________________________________

# Recalculate Inertia
inertia = kmeans_6.inertia_
print(f"Inertia for 6 clusters before outlier removal: {inertia}")

# Recalculate Silhouette Score
silhouette_avg = silhouette_score(scaled_features, kmeans_6.labels_)
print(f"Silhouette Score for 6 clusters before outlier removal: {silhouette_avg}")

# Calculate the Euclidean distance of each point to its cluster centroid
def calculate_distances_to_centroids(df, centroids, labels):
    distances = []
    for i in range(features.shape[0]):  # Iterate over each point
        centroid = centroids[labels[i]]  # Get the centroid corresponding to the cluster label
        point = scaled_features[i]  # Get the scaled data point
        distance = np.linalg.norm(point - centroid)  # Calculate Euclidean distance
        distances.append(distance)
    return distances

# Assuming df_no_outliers is the DataFrame after removing global outliers
centroids = kmeans_6.cluster_centers_
distances = calculate_distances_to_centroids(scaled_features, centroids, kmeans_6.labels_)
df['Distance_to_Centroid'] = distances

# Define the threshold for outliers (e.g., top 5% of distances)
threshold = df['Distance_to_Centroid'].quantile(0.95)

# Remove outliers
df_cleaned = df[df['Distance_to_Centroid'] <= threshold]

# Check how many data points were removed
print(f"Data reduced from {df.shape[0]} to {df_cleaned.shape[0]} rows after removing cluster outliers.")

# Reapply K-means clustering after removing outliers within clusters
scaled_features_cleaned = scaler.fit_transform(df_cleaned[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
kmeans_cleaned = KMeans(n_clusters=6, random_state=42)
df_cleaned['Cluster_6'] = kmeans_cleaned.fit_predict(scaled_features_cleaned)

# Visualize the new clusters
plt.figure(figsize=(10, 8))
for cluster in range(6):
    plt.scatter(
        df_cleaned[df_cleaned['Cluster_6'] == cluster]['Annual Income (k$)'],
        df_cleaned[df_cleaned['Cluster_6'] == cluster]['Spending Score (1-100)'],
        label=f'Cluster {cluster}'
    )
centroids_cleaned = kmeans_cleaned.cluster_centers_
plt.scatter(
    centroids_cleaned[:, 1] * scaler.scale_[1] + scaler.mean_[1],  
    centroids_cleaned[:, 2] * scaler.scale_[2] + scaler.mean_[2],  
    s=300, c='black', marker='x', label='Centroids'
)
plt.title('Customer Segmentation After Removing Cluster Outliers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Recalculate Inertia
inertia = kmeans_cleaned.inertia_
print(f"Inertia for 6 clusters after outlier removal: {inertia}")

# Recalculate Silhouette Score
silhouette_avg = silhouette_score(scaled_features_cleaned, kmeans_cleaned.labels_)
print(f"Silhouette Score for 6 clusters after outlier removal: {silhouette_avg}")