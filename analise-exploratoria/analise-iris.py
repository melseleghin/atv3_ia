import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Carregar a base de dados Iris - com rotulos 
iris, iris_target = load_iris(return_X_y=True)
iris_df = pd.DataFrame(iris, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

# Adicionando os rótulos das classes (espécies)
iris_df['species'] = iris_target

# 1. Exploração da base
num_samples, num_features = iris.shape
print(f"Número de amostras: {num_samples}")
print(f"Número de atributos: {num_features}")

# Verificar valores faltantes
missing_values = iris_df.isnull().sum()
print("\nValores ausentes:")
print(missing_values)

# Estatísticas básicas
print("\nEstatísticas básicas (describe()):")
print(iris_df.describe())

# 2. Visualização com gráficos

# Histogramas das características
iris_df.drop(columns='species').hist(bins=20, edgecolor='black', figsize=(10, 8), layout=(2, 2))
plt.suptitle('Histogramas das Características - Iris')
plt.show()

# Boxplots para detectar outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=iris_df.drop(columns='species'))
plt.title('Boxplots das Características - Iris')
plt.show()

# 3. Matriz de Correlação
correlation_matrix = iris_df.drop(columns='species').corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação - Iris')
plt.show()

# 4. Scatterplot 2D para visualização
plt.figure(figsize=(8, 6))
sns.scatterplot(x=iris_df['sepal_length'], y=iris_df['sepal_width'], hue=iris_df['species'], palette='viridis')
plt.title('Scatterplot 2D - Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# 5. PCA (Principal Component Analysis) para redução de dimensões e visualização das classes
pca = PCA(n_components=2)
pca_result = pca.fit_transform(iris_df.drop(columns='species'))

# Criando um DataFrame com os componentes principais
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['species'] = iris_df['species']

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df, palette='viridis')
plt.title('PCA - Separação das Classes')
plt.show()
