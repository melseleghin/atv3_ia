import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, single, complete, average, ward


wine, _ = load_wine(return_X_y=True)
num_samples, num_features = wine.shape
Z = single(wine)

plt.figure(figsize=(min(num_samples / 20, 15), 5))
dendrogram(Z, truncate_mode='level', p=10)
plt.title('Single Linkage Dendrogram - Wine')
plt.show()

clusters = fcluster(Z, t=4, criterion='maxclust')
plt.scatter(wine[:, 0], wine[:, 1], c=clusters, cmap="viridis")
plt.gca().set_aspect('auto')
plt.title('Single Linkage Clustering - Wine')
plt.show()

#---------------------------------------------------------------------------

Z = complete(wine)

plt.figure(figsize=(min(num_samples / 20, 15), 5))
dendrogram(Z, truncate_mode='level', p=10)
plt.title('Complete Linkage Dendrograma - Wine')
plt.show()

clusters = fcluster(Z, t=4, criterion='maxclust')
plt.scatter(wine[:, 0], wine[:, 1], c=clusters, cmap="viridis")
plt.gca().set_aspect('auto')
plt.title('Complete Linkage Clustering - Wine')
plt.show()

#---------------------------------------------------------------------------


Z = average(wine)

plt.figure(figsize=(min(num_samples / 20, 15), 5))
dendrogram(Z, truncate_mode='level', p=10)
plt.title('Average Linkage Dendrograma - Wine')
plt.show()

clusters = fcluster(Z, t=4, criterion='maxclust')
plt.scatter(wine[:, 0], wine[:, 1], c=clusters, cmap="viridis")
plt.gca().set_aspect('auto')
plt.title('Average Linkage Clustering - Wine')
plt.show()

#---------------------------------------------------------------------------

Z = ward(wine)

plt.figure(figsize=(min(num_samples / 20, 15), 5))
dendrogram(Z, truncate_mode='level', p=10)
plt.title('Ward Linkage Dendrograma - Wine')
plt.show()

clusters = fcluster(Z, t=4, criterion='maxclust')
plt.scatter(wine[:, 0], wine[:, 1], c=clusters, cmap="viridis")
plt.gca().set_aspect('auto')
plt.title('Ward Linkage Clustering - Wine')
plt.show()
