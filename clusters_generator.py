from typing import *
import numpy as np

class clusters_generator:

    def __init__(
        self,
        n: int,
        d: int,
        a: int,
        b: int,
        cluster_std: int,
        num_clusters: int,
        n_test_per_cluster: int,
    ):
        self.n = n
        self.d = d
        self.a = a
        self.b = b
        self.cluster_std = cluster_std
        self.num_clusters = num_clusters
        self.n_test_per_cluster = n_test_per_cluster

    def _create_one_cluster(self, n_par: int, d: int, a: int, b: int, std: float = 1):
        rand_mean = np.random.uniform(high=b, low=a, size=(1, d))
        return np.random.normal(loc=0, scale=std, size=(n_par, d)) + rand_mean, rand_mean

    def _create_one_cluster_with_mean(self, n_per_cluster: int, d: int, std: float, mean: np.ndarray):
        return np.random.normal(loc=0, scale=std, size=(n_per_cluster, d)) + mean

    def _create_cluster(self, k: int, n: int, d: int, a: int, b: int, std: float = 1):
        means = np.random.uniform(high=b, low=a, size=(k, d))
        clusters = np.random.normal(loc=0, scale=std, size=(n, d)) + means.repeat(repeats=n // k, axis=0)
        return clusters, means

    def _create_test_cluster(self, clusters_mean: np.ndarray, n_per_cluster: int, std):
        n, d = clusters_mean.shape
        clusters_test = np.random.normal(loc=0, scale=std, size=(n * n_per_cluster, d)) + clusters_mean.repeat(repeats=n_per_cluster, axis=0)
        labels = np.arange(n).repeat(repeats=n_per_cluster)
        return clusters_test, labels.flatten()

    def _create_test_data(self):
        print("===== Creating Training Cluster =====")
        clusters_train, clusters_means = self._create_cluster(self.num_clusters, self.n, self.d, self.a, self.b, self.cluster_std)
        print("===== Creating Test Cluster =====")
        clusters_test, labels = self._create_test_cluster(clusters_means, self.n_test_per_cluster, self.cluster_std)
        return clusters_train, clusters_means, clusters_test, labels

    def generate(self):
        return self._create_test_data()