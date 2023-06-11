from typing import *
import numpy as np
from performance_test_data import *

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
        sparsity: float
    ):
        self.n = n
        self.d = d
        self.a = a
        self.b = b
        self.cluster_std = cluster_std
        self.num_clusters = num_clusters
        self.n_test_per_cluster = n_test_per_cluster
        self.sparsity = sparsity
        self.characteristics = {
            "n": n,
            "d": d,
            "a": a,
            "b": b,
            "cluster_std": cluster_std,
            "num_clusters": num_clusters,
            "sparsity": sparsity
        }

    def _create_one_cluster(self, n_par: int, d: int, a: int, b: int, std: float = 1):
        rand_mean = np.random.uniform(high=b, low=a, size=(1, d))
        return np.random.normal(loc=0, scale=std, size=(n_par, d)) + rand_mean, rand_mean

    def _create_one_cluster_with_mean(self, n_per_cluster: int, d: int, std: float, mean: np.ndarray):
        return np.random.normal(loc=0, scale=std, size=(n_per_cluster, d)) + mean

    def _create_one_sparse_mask(self):
        return np.random.binomial(1, np.array([1 - self.sparsity]).repeat(repeats=self.d))

    def _create_sparse_mask(self) -> np.ndarray:
        all_masks = np.array(list(map(lambda _: self._create_one_sparse_mask(), [1 for i in range(self.num_clusters)])))
        return all_masks

    def _create_cluster(self, k: int, n: int, d: int, a: int, b: int, std: float = 1):
        means = np.random.uniform(high=b, low=a, size=(k, d))
        clusters = np.random.normal(loc=0, scale=std, size=(n, d)) + means.repeat(repeats=n // k, axis=0)
        return clusters, means

    def _create_test_cluster(self, clusters_mean: np.ndarray, n_per_cluster: int, std):
        n, d = clusters_mean.shape
        clusters_test = np.random.normal(loc=0, scale=std, size=(n * n_per_cluster, d)) + clusters_mean.repeat(repeats=n_per_cluster, axis=0)
        labels = np.arange(n).repeat(repeats=n_per_cluster)
        return clusters_test, labels.flatten()

    def _create_test_data(self) -> performance_test_data:
        print("===== Creating Mask =====")
        all_masks: np.ndarray = self._create_sparse_mask()
        print("===== Creating Training Cluster =====")
        clusters_train, clusters_means = self._create_cluster(self.num_clusters, self.n, self.d, self.a, self.b, self.cluster_std)
        print("===== Creating Test Cluster =====")
        clusters_test, labels = self._create_test_cluster(clusters_means, self.n_test_per_cluster, self.cluster_std)

        clusters_train_masked = np.multiply(clusters_train, all_masks.repeat(repeats=self.n//self.num_clusters, axis=0))
        clusters_mean_masked = np.multiply(clusters_means, all_masks)
        clusters_test_masked = np.multiply(clusters_test, all_masks.repeat(repeats=self.n_test_per_cluster, axis=0))

        return performance_test_data(clusters_train_masked, clusters_mean_masked, clusters_test_masked, labels, self.characteristics)

    def generate(self) -> performance_test_data:
        return self._create_test_data()