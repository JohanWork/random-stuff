import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Convert embeddings to a numpy array
embeddings_array = np.array(dataset['embeddings'])

# Perform K-means clustering (adjust n_clusters as needed)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings_array)

# Reduce dimensionality for visualization using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_array)

# Create a colormap
colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

# Plot the clusters
plt.figure(figsize=(10, 8))
for i in range(n_clusters):
    mask = cluster_labels == i
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
               c=[colors[i]], label=f'Cluster {i}',
               alpha=0.6)

plt.title('Text Clusters based on ModernBERT Embeddings')
plt.legend()
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.show()

# Add cluster labels to the dataset
dataset = dataset.add_column("cluster", cluster_labels)

# Print a few examples from each cluster
for i in range(n_clusters):
    print(f"\nCluster {i} examples:")
    cluster_examples = dataset.filter(lambda x: x['cluster'] == i)
    for j, example in enumerate(cluster_examples[:3]['anchor']):
        print(f"{j+1}. {example}")
