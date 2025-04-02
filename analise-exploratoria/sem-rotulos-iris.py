import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans  # Importar o KMeans

# Carregar a base de dados Iris (sem as classes, apenas as features)
iris, _ = load_iris(return_X_y=True)

# Exploração da base
num_samples, num_features = iris.shape
print(f"Número de amostras: {num_samples}")
print(f"Número de atributos: {num_features}")

# Visualização com gráficos

# Histogramas das características
plt.figure(figsize=(10, 8))
plt.hist(iris, bins=20, edgecolor='black', histtype='bar')
plt.suptitle('Histogramas das Características - Iris')
plt.show()

# Boxplots para detectar outliers
plt.figure(figsize=(12, 6))
plt.boxplot(iris)
plt.title('Boxplots das Características - Iris')
plt.show()

# Matriz de correlação (heatmap)
correlation_matrix = np.corrcoef(iris, rowvar=False)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação - Iris')
plt.show()

# Aplicando o K-means para gerar clusters
kmeans = KMeans(n_clusters=3)  # Definindo 3 clusters (pois o Iris tem 3 classes)
kmeans.fit(iris)
clusters = kmeans.labels_  # Obtendo os rótulos dos clusters

# Scatterplot 2D com clusters
plt.figure(figsize=(8, 6))
plt.scatter(iris[:, 0], iris[:, 1], c=clusters, cmap="viridis")  # Agora 'clusters' é definido
plt.title('Scatterplot 2D - Sepal Length vs Sepal Width (Com Clusters)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
