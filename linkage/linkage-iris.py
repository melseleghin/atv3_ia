import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, single, complete, average, ward


iris, _ = load_iris(return_X_y=True)
num_samples, num_features = iris.shape
Z = single(iris)

plt.figure(figsize=(min(num_samples / 20, 15), 5))
dendrogram(Z, truncate_mode='level', p=10)
plt.title('Single Linkage Dendrogram - Iris')
plt.show()

clusters = fcluster(Z, t=3, criterion='maxclust')
plt.scatter(iris[:, 0], iris[:, 1], c=clusters, cmap="viridis")
plt.gca().set_aspect('auto')
plt.title('Single Linkage Clustering - Iris')
plt.show()

#---------------------------------------------------------------------------

Z = complete(iris)

plt.figure(figsize=(min(num_samples / 20, 15), 5))
dendrogram(Z, truncate_mode='level', p=10)
plt.title('Complete Linkage Dendrograma - Iris')
plt.show()

clusters = fcluster(Z, t=3, criterion='maxclust')
plt.scatter(iris[:, 0], iris[:, 1], c=clusters, cmap="viridis")
plt.gca().set_aspect('auto')
plt.title('Complete Linkage Clustering - Iris')
plt.show()

#---------------------------------------------------------------------------


Z = average(iris)

plt.figure(figsize=(min(num_samples / 20, 15), 5))
dendrogram(Z, truncate_mode='level', p=10)
plt.title('Average Linkage Dendrograma - Iris')
plt.show()

clusters = fcluster(Z, t=3, criterion='maxclust')
plt.scatter(iris[:, 0], iris[:, 1], c=clusters, cmap="viridis")
plt.gca().set_aspect('auto')
plt.title('Average Linkage Clustering - Iris')
plt.show()

#---------------------------------------------------------------------------

Z = ward(iris)

plt.figure(figsize=(min(num_samples / 20, 15), 5))
dendrogram(Z, truncate_mode='level', p=10)
plt.title('Ward Linkage Dendrograma - Iris')
plt.show()

clusters = fcluster(Z, t=3, criterion='maxclust')
plt.scatter(iris[:, 0], iris[:, 1], c=clusters, cmap="viridis")
plt.gca().set_aspect('auto')
plt.title('Ward Linkage Clustering - Iris')
plt.show()
