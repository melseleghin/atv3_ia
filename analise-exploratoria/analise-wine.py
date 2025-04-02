import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA


wine, wine_target = load_wine(return_X_y=True)
wine_df = pd.DataFrame(wine, columns=["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", 
                                      "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", 
                                      "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"])


wine_df['wine_type'] = wine_target


num_samples, num_features = wine.shape
print(f"Número de amostras: {num_samples}")
print(f"Número de atributos: {num_features}")


missing_values = wine_df.isnull().sum()
print("\nValores ausentes:")
print(missing_values)


print("\nEstatísticas básicas (describe()):")
print(wine_df.describe())


plt.figure(figsize=(12, 10)) 
wine_df.drop(columns='wine_type').hist(bins=20, edgecolor='black', figsize=(10, 8))
plt.suptitle('Histogramas das Características - Wine')
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=wine_df.drop(columns='wine_type'))
plt.title('Boxplots das Características - Wine')
plt.show()


correlation_matrix = wine_df.drop(columns='wine_type').corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação - Wine')
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(x=wine_df['alcohol'], y=wine_df['malic_acid'], hue=wine_df['wine_type'], palette='viridis')
plt.title('Scatterplot 2D - Alcohol vs Malic Acid')
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.show()


pca = PCA(n_components=2)
pca_result = pca.fit_transform(wine_df.drop(columns='wine_type'))

pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['wine_type'] = wine_df['wine_type']

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='wine_type', data=pca_df, palette='viridis')
plt.title('PCA - Separação das Classes - Wine')
plt.show()
