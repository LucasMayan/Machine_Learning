import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Gerar dados de exemplo
np.random.seed(42)
X = np.random.rand(300, 2)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotar os resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
plt.title('Clustering com K-Means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
