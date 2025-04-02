import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans  

iris, _ = load_iris(return_X_y=True)

num_samples, num_features = iris.shape
print(f"Número de amostras: {num_samples}")
print(f"Número de atributos: {num_features}")


plt.figure(figsize=(10, 8))
plt.hist(iris, bins=20, edgecolor='black', histtype='bar')
plt.suptitle('Histogramas das Características - Iris')
plt.show()

plt.figure(figsize=(12, 6))
plt.boxplot(iris)
plt.title('Boxplots das Características - Iris')
plt.show()

correlation_matrix = np.corrcoef(iris, rowvar=False)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação - Iris')
plt.show()

kmeans = KMeans(n_clusters=3)  
kmeans.fit(iris)
clusters = kmeans.labels_  

plt.figure(figsize=(8, 6))
plt.scatter(iris[:, 0], iris[:, 1], c=clusters, cmap="viridis")  
plt.title('Scatterplot 2D - Sepal Length vs Sepal Width (Com Clusters)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
