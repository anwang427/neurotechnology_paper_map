import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import mplcursors

# Load the preprocessed and selected data from the CSV file
selected_data = pd.read_csv('preprocessed_neurotechnology_papers.csv')

# Extract the selected features and the citations column
features = selected_data.drop(['Citations', 'Title'], axis=1).values

# Apply dimensionality reduction using PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Apply K-means clustering on the feature set
k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(reduced_features)

# Add the cluster labels to the DataFrame
selected_data['Cluster'] = cluster_labels

# Calculate the mean TF-IDF score for each feature within each cluster
mean_tfidf = selected_data.groupby('Cluster').mean()

# Get the top words for each cluster
n_words = 10  # Number of top words to consider
top_words_per_cluster = []
cluster_labels_legend = []  # Store cluster labels for legend
for cluster in range(k):
    top_words = mean_tfidf.loc[cluster].nlargest(n_words).index.tolist()
    top_words_per_cluster.append(top_words)
    cluster_label = f"Cluster {cluster+1}: {', '.join(top_words)}"
    cluster_labels_legend.append(cluster_label)

# Visualize the clusters with interactive tooltips
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each cluster separately and assign cluster labels as legend labels
scatter = None
for cluster in range(k):
    cluster_data = reduced_features[cluster_labels == cluster]
    scatter = ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=cluster_labels_legend[cluster])

# Create tooltips for cluster words
if scatter is not None:
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(selected_data['Title'][sel.target.index]))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Neurotechnology Papers Clusters')
plt.legend()
plt.show()
