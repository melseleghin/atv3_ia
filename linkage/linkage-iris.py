import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, single, complete, average, ward


iris, _ = load_iris(return_X_y=True)
wine, _ = load_wine(return_X_y=True)

Z_iris = single(iris)

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Single Linkage Dendrogram')
plt.show()

clusters = fcluster(Z, t=3, criterion='maxclust')
plt.scatter(iris[:, 0], iris[:, 1], c=clusters, cmap="viridis")
plt.title('Single Linkage Clustering')
plt.show()

#---------------------------------------------------------------------------

Z = complete(iris)

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Complete Linkage Dendrograma')
plt.show()

clusters = fcluster(Z, t=4, criterion='maxclust')
plt.scatter(iris[:, 0], iris[:, 1], c=clusters, cmap="viridis")
plt.title('Complete Linkage Clustering')
plt.show()

#---------------------------------------------------------------------------


Z = average(iris)

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Average Linkage Dendrograma')
plt.show()

clusters = fcluster(Z, t=3, criterion='maxclust')
plt.scatter(iris[:, 0], iris[:, 1], c=clusters, cmap="viridis")
plt.title('Average Linkage Clustering')
plt.show()

#---------------------------------------------------------------------------

Z = ward(iris)

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Ward Linkage Dendrograma')
plt.show()

clusters = fcluster(Z, t=3, criterion='maxclust')
plt.scatter(iris[:, 0], iris[:, 1], c=clusters, cmap="viridis")
plt.title('Ward Linkage Clustering')
plt.show()
