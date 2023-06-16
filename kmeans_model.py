import sklearn as sk
import sklearn.cluster

from performance_test_model import *
import sklearn as sk
import sklearn.cluster

from performance_test_model import *


class kmeans_model(performance_test_model):

    def __init__(self):
        self.clusters_train = None
        self.clusters_means = None
        self.clusters_test = None
        self.clusters_test_label = None

        self.n = None
        self.num_cluster = None
        self.n_test_per_cluster = None
        self.model: sklearn.cluster.KMeans = None

        self.reduc_clusters_train = None
        self.reduc_clusters_means = None
        self.reduc_clusters_test = None

    def initialize(self, data: performance_test_data):
        self.clusters_train = data.training_data
        self.clusters_means = data.training_label
        self.clusters_test = data.testing_data
        self.clusters_test_label = data.testing_label

        self.n = self.clusters_train.shape[0]
        self.num_cluster = self.clusters_means.shape[0]
        self.n_test_per_cluster = self.clusters_test.shape[0] // self.num_cluster
        self.model: sklearn.cluster.KMeans = sk.cluster.KMeans(n_clusters=self.num_cluster)

        self.reduc_clusters_train = None
        self.reduc_clusters_means = None
        self.reduc_clusters_test = None

    def apply_dim_reduc(self, dim_reduc_f) -> np.ndarray:
        reduc = dim_reduc_f(np.concatenate((self.clusters_train, self.clusters_test, self.clusters_means)))
        self.reduc_clusters_train = reduc[:self.n, :]
        self.reduc_clusters_test = reduc[self.n:self.n + self.clusters_test.shape[0],:]
        self.reduc_clusters_means = reduc[self.n + self.clusters_test.shape[0]:, :]
        return reduc

    def train(self):
        self.model.fit(self.reduc_clusters_train)

    def compute_accuracy(self):
        predicted = self.model.predict(self.reduc_clusters_test)
        predicted_actual = self.model.predict(self.reduc_clusters_means).repeat(repeats=self.n_test_per_cluster)
        return np.sum(predicted == predicted_actual) / predicted.shape[0]
