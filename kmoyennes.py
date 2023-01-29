import math, numpy, pandas
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris


K = 3
MAX = 150
# iris_data = pandas.read_csv(
#     "./data/iris_data.csv", sep=";", index_col=False, header=None
# )
iris_labels = numpy.genfromtxt(
    "./data/iris_label.csv",
)
iris_data = numpy.genfromtxt(
    "./data/iris_data.csv", delimiter=";", dtype=float, filling_values=0
)

print(iris_data.shape)


def distance(x, y):
    return numpy.sqrt(numpy.sum((x - y) ** 2))


class kmeans:
    def __init__(self, k=3, max_iter=100, plot_steps=False):
        self.k = k
        self.max_iter = max_iter
        self.plot_steps = plot_steps
        # list de list
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def predict(self, x):
        self.x = x
        # n_samples = longueur du fichier, n_features = donnée largeur
        self.n_samples, self.n_features = x.shape
        # Init
        rand_k = numpy.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.x[i] for i in rand_k]
        # opti
        for _ in range(self.max_iter):
            # update clusters
            self.clusters = self.__create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            # update centroids
            centroids_old = self.centroids
            self.centroids = self.__get_centroids(self.clusters)
            if self.plot_steps:
                self.plot()
            # conversion needed
            if self.__is_converged(centroids_old, self.centroids):
                break
        # classicfication return labels
        return self.__get_cluster_labels(self.clusters)

    def __create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for i, sample in enumerate(self.x):
            centroid_i = self.__closest_centroid(sample, centroids)
            clusters[centroid_i].append(i)
        return clusters

    def __closest_centroid(self, sample, centroids):
        distances = [distance(sample, point) for point in centroids]
        closest_i = numpy.argmin(distances)
        return closest_i

    def __get_centroids(self, clusters):
        centroids = numpy.zeros((self.k, self.n_features))
        for cluster_i, cluster in enumerate(clusters):
            clusters_mean = numpy.mean(self.x[cluster], axis=0)
            centroids[cluster_i] = clusters_mean
        return centroids

    def __is_converged(self, old, new):
        distances = [distance(old[i], new[i]) for i in range(self.k)]
        return sum(distances) == 0

    def __get_cluster_labels(self, clusters):
        labels = numpy.empty(self.n_samples)
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_i
        return labels

    def plot(self):
        (
            fig,
            ax,
        ) = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.x[index].T
            ax.scatter(*point)

        legend_labels = []
        for i in range(len(self.centroids)):
            legend_labels.append("[{}]".format(i))

        for point in self.centroids:
            ax.scatter(
                *point, marker="x", color="black", linewidth=2, label=legend_labels
            )

        plt.legend()
        plt.show()


X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
# print(x)
iris = load_iris()
x = iris.data[:, :2]
y = iris_labels
kmeans = kmeans(k=K, max_iter=MAX, plot_steps=True)
data = kmeans.predict(x)
