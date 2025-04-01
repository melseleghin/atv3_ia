import matplotlib.pyplot as plt
from sklearn.cluster import BisectingKMeans
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA


class BisectingKMeansAnalytic:

    def __init__(self, dataset):
        self.dataset = dataset
        self.reduced_data = None
        self.list_clusters = []
        self.list_inertias = []
        self.qtd_clusters_otima = 0
        self.target = None

    def bisecting_kmeans(self, qtd_clusters):
        return BisectingKMeans(n_clusters=qtd_clusters).fit(self.dataset.data)

    @staticmethod
    def inercia(model):
        return model.inertia_

    def run_bisecting_kmeans_with_dif_knumbers(self):
        for i in range(1, 11):
            self.list_clusters.append(i)
            model = self.bisecting_kmeans(i)
            inertia = self.inercia(model)
            self.list_inertias.append(inertia)

    def plot_elbow_method(self):
        plt.plot(self.list_clusters, self.list_inertias)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method with inertia (Bisecting K-Means)')
        plt.xticks(self.list_clusters)
        plt.grid()
        plt.show()

    def apply_pca(self):
        model = BisectingKMeans(n_clusters=self.qtd_clusters_otima).fit(self.dataset.data)
        self.target = model.labels_
        self.reduced_data = PCA(n_components=2).fit_transform(self.dataset.data)

    def plot_bisecting_kmeans_clustering(self):
        fig = plt.figure(1, figsize=(10, 8))
        ax = fig.add_subplot()
        ax.scatter(
            self.reduced_data[:, 0],
            self.reduced_data[:, 1],
            c=self.target,
            s=40
        )
        plt.show()


if __name__ == "__main__":
    # PARA BASE IRIS:
    print("### BASE DE DADOS IRIS ###")
    iris = load_iris()
    bkm_iris = BisectingKMeansAnalytic(iris)
    bkm_iris.run_bisecting_kmeans_with_dif_knumbers()
    bkm_iris.plot_elbow_method()
    bkm_iris.qtd_clusters_otima = int(input("Qual o número de k ótimo apontado pelo elbow method? (no nosso caso, é 3):  "))
    print(f"Plotando gráfico com {bkm_iris.qtd_clusters_otima} clusters...")
    bkm_iris.apply_pca()
    bkm_iris.plot_bisecting_kmeans_clustering()

    # PARA BASE WINE:
    print("### BASE DE DADOS WINE ###")
    wine = load_wine()
    bkm_wine = BisectingKMeansA
    nalytic(wine)
    bkm_wine.run_bisecting_kmeans_with_dif_knumbers()
    bkm_wine.plot_elbow_method()
    bkm_wine.qtd_clusters_otima = int(input("Qual o número de k ótimo apontado pelo elbow method? (no nosso caso, é 3):  "))
    print(f"Plotando gráfico com {bkm_wine.qtd_clusters_otima} clusters...")
    bkm_wine.apply_pca()
    bkm_wine.plot_bisecting_kmeans_clustering()
