import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, single, complete, average, ward

wine, _ = load_wine(return_X_y=True)
num_samples, num_features = wine.shape  


def elbow_method(data, max_clusters=10):
    """
    Aplica o método do cotovelo para determinar o número ideal de clusters.
    Parâmetros:
        data (array-like): Dados a serem agrupados.
        max_clusters (int): Número máximo de clusters a ser considerado.
    Retorna:
        int: Número ótimo de clusters baseado na análise da inércia.
    """
    inertia = []  
    cluster_range = range(2, max_clusters + 1)  

    for k in cluster_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(data)
        inertia.append(model.inertia_)  

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertia, marker='o', linestyle='-')
    plt.xlabel('Clusters')
    plt.ylabel('Inércia')
    plt.title('Elbow Method')
    plt.show()

    return cluster_range[np.argmin(np.diff(inertia))]



optimal_k = elbow_method(wine)
linkage_methods = ["single", "complete", "average", "ward"]

for method in linkage_methods:
    Z = linkage(wine, method=method)
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode='level', p=10)
    plt.title(f'Dendrograma - {method.capitalize()} Linkage')
    plt.show()
    
    clusters = fcluster(Z, t=optimal_k, criterion='maxclust')

    plt.scatter(wine[:, 0], wine[:, 1], c=clusters, cmap='viridis')
    plt.title(f'Clustering - {method.capitalize()} Linkage')
    plt.show()
