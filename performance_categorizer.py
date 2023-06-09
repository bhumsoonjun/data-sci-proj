import numpy as np
import sklearn as sk
import scipy as sc
from typing import *
import time

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from jlt.jlt import *
from dataclasses import dataclass
from performance_test_data import *
from stats import stats


class performance_cat:

    def __init__(self, data_tag: performance_test_data, num_test):
        self.data_tag = data_tag
        self.training_data = data_tag.training_data
        self.training_label = data_tag.training_label
        self.testing_data = data_tag.testing_data
        self.testing_label = data_tag.testing_label
        self.num_test = num_test

    def _rescale(self, data: np.ndarray):
        stds = data.std(axis=0)
        return sc.cluster.vq.whiten(data), stds

    def _reverse_rescale(self, data: np.ndarray, stds: np.ndarray):
        return data * stds

    def _apply_dim_reduc(self, f, data):
        return f(data)

    def _time_proc(self, f):
        start = time.perf_counter_ns()
        res = f()
        return res, (time.perf_counter_ns() - start) / 1e9

    def _find_min_indices(self, point: np.ndarray, all_clusters):
        dir = point - all_clusters
        dist = np.sum(dir * dir, axis=1)
        return np.argmin(dist)

    def _compute_accuracy(self, predicted: np.ndarray, predicted_actual: np.ndarray, n_test_per_cluster: int):
        return np.sum(predicted == predicted_actual.repeat(repeats=n_test_per_cluster)) / predicted.shape[0]

    def _performance_test_one_func(
            self,
            clusters_train: np.ndarray,
            clusters_means: np.ndarray,
            clusters_test: np.ndarray,
            num_test: int,
            n_test_per_cluster: int,
            dim_reduc_f
    ):
        all_accuracy = []
        all_dim_reduc_time = []
        all_train_time = []
        all_trans_stats = []
        for i in range(num_test):
            print(f"==== Iter {i} ====")
            print("=== Applying Dimensionality Reduction ===")
            reduc, dim_reduc_time = self._time_proc(lambda: self._apply_dim_reduc(dim_reduc_f, np.concatenate((clusters_train, clusters_test, clusters_means))))
            reduc_train, reduc_test, reduc_means = reduc[:self.n, :], reduc[self.n:self.n + clusters_test.shape[0], :], reduc[self.n + clusters_test.shape[0]:,:]
            print("=== Traning Model ===")
            model, train_time = self._time_proc(lambda: KMeans(n_clusters=self.num_clusters).fit(reduc_train))
            print("=== Predicting Model ===")
            predicted = model.predict(reduc_test)
            predicted_actual = model.predict(reduc_means)
            print("=== Computing Accuracy ===")
            all_accuracy.append(self._compute_accuracy(predicted, predicted_actual, n_test_per_cluster))
            all_dim_reduc_time.append(dim_reduc_time)
            all_train_time.append(train_time)
            all_trans_stats.append(self._get_data_stats(reduc_train))
        return all_accuracy, all_dim_reduc_time, all_train_time, all_trans_stats

    def performance_test_all(self, names, dim_reduc_f, params):
        clusters_train, clusters_means, clusters_test, labels = self._create_test_data()
        ori_stds_sum, ori_stds_mean, ori_shape = self._get_data_stats(clusters_train)
        result = []
        for name, func, param in zip(names, dim_reduc_f, params):
            all_accuracy, all_dim_reduc_time, all_train_time, all_trans_stats = self._performance_test_one_func(clusters_train, clusters_means, clusters_test, self.num_test, self.n_test_per_cluster, func)
            four_ways = zip(all_accuracy, all_dim_reduc_time, all_train_time, all_trans_stats)
            for tup in four_ways:
                acc, reduc_time, train_time, trans_stats = tup
                trans_stds_sum, trans_stds_mean, trans_shape = trans_stats
                stat = stats(name, param, reduc_time, train_time, acc, ori_stds_sum, ori_stds_mean, ori_shape, trans_stds_sum, trans_stds_mean, trans_shape, self.num_clusters, self.a, self.b, self.cluster_std)
                result.append(stat)
        return result

