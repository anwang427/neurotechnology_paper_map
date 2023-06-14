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
for cluster in range(k):
    top_words = mean_tfidf.loc[cluster].nlargest(n_words).index.tolist()
    top_words_per_cluster.append(top_words)

# Print the top words for each cluster
for cluster, words in enumerate(top_words_per_cluster):
    print(f"Cluster {cluster+1} - Top Words: {', '.join(words)}")

# Visualize the clusters with interactive tooltips
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels)

# Create tooltips for cluster words
mplcursors.cursor(scatter).connect("add", lambda sel: sel.annotation.set_text(selected_data['Title'][sel.target.index]))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Neurotechnology Papers Clusters')
plt.show()
