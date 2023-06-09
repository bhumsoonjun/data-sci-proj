from typing import *
import numpy as np
import sklearn as sk
from performance_test_data import *

class kmeans_performance_test:

    def __init__(
            self,
            data: performance_test_data
    ):
        self.clusters_train = data.training_data
        self.clusters_means = data.training_label
        self.clusters_test = data.testing_data
        self.clusters_test_label = data.testing_label

        self.num_cluster = self.clusters_means.shape[0]
        self.model = sk.cluster.KMeans(n_clusters=self.num_cluster.shape)

    def apply_dim_reduc(self, dim_reduc_f):
        return dim_reduc_f(np.concatenate((self.clusters_train, self.clusters_test, self.clusters_means)))

    def train(self):
        return self.model.fit_transform(self.clusters_train)

    def _compute_accuracy(self, predicted: np.ndarray, predicted_actual: np.ndarray, n_test_per_cluster: int):
        return np.sum(predicted == predicted_actual.repeat(repeats=n_test_per_cluster)) / predicted.shape[0]

    def test(self):
        pass