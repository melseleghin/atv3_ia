import matplotlib.pyplot as plt
from sklearn.cluster import BisectingKMeans
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA


class BisectingKMeansAnalytic:
    def __init__(self, dataset):
        self.dataset = dataset
        self.reduced_data = None
        self.qtd_clusters_otima = 2   
        self.target = None

    def bisecting_kmeans(self):
        return BisectingKMeans(n_clusters=self.qtd_clusters_otima).fit(self.dataset.data)

    def apply_pca(self):
        model = self.bisecting_kmeans()
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
        plt.title(f'Bisecting K-Means Clustering (k={self.qtd_clusters_otima})')
        plt.show()


if __name__ == "__main__":
    # PARA BASE IRIS:
    print("### BASE DE DADOS IRIS ###")
    iris = load_iris()
    bkm_iris = BisectingKMeansAnalytic(iris)
    bkm_iris.apply_pca()
    bkm_iris.plot_bisecting_kmeans_clustering()

    # PARA BASE WINE:
    print("### BASE DE DADOS WINE ###")
    wine = load_wine()
    bkm_wine = BisectingKMeansAnalytic(wine)
    bkm_wine.apply_pca()
    bkm_wine.plot_bisecting_kmeans_clustering()
