import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA



wine, _ = load_wine(return_X_y=True)


num_samples, num_features = wine.shape
print("Número de amostras e atributos:", (num_samples, num_features))


mean_values = np.mean(wine, axis=0)
std_values = np.std(wine, axis=0)
print("\nMédia das características:", mean_values)
print("\nDesvio padrão das características:", std_values)


print("\nValores ausentes: Nenhum valor ausente foi identificado, pois os dados estão completos no scikit-learn.")


plt.figure(figsize=(12, 10))
plt.hist(wine, bins=20, edgecolor='black', histtype='bar', label=[f'Feature {i+1}' for i in range(num_features)])
plt.title('Histograma das Características do Vinho')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
sns.boxplot(data=wine, orient="h")
plt.title('Boxplot das Características do Vinho')
plt.show()

plt.figure(figsize=(12, 10))
correlation_matrix = np.corrcoef(wine, rowvar=False)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação - Wine')
plt.show()

pca = PCA(n_components=2)
wine_pca = pca.fit_transform(wine)

plt.figure(figsize=(10, 8))
plt.scatter(wine_pca[:, 0], wine_pca[:, 1], c='blue')
plt.title('Scatterplot 2D - PCA das Características do Vinho')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

