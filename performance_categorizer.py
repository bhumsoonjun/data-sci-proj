import numpy as np
import sklearn as sk
import scipy as sc
from typing import *
import time
from sklearn.cluster import KMeans
from performance_test_data import *
from stats import stats
from performance_test_model import *
from dim_reduc_function import *
from utils import *
from data_stats import  *


class performance_cat:

    def __init__(self, num_test: int):
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

    def _performance_test_one_func(
            self,
            data: performance_test_data,
            model: performance_test_model,
            dim_reduc_f: dim_reduc_function
    ):
        all_accuracy = []
        all_dim_reduc_time = []
        all_train_time = []
        all_trans_stats = []

        model.initialize(data)

        for i in range(self.num_test):
            print(f"==== Iter {i} ====")
            print("=== Applying Dimensionality Reduction ===")

            reduc, dim_reduc_time = self._time_proc(lambda: model.apply_dim_reduc(dim_reduc_f.f))

            print("=== Traning Model ===")
            _, train_time = self._time_proc(lambda: model.train())

            print("=== Computing Accuracy ===")
            all_accuracy.append(model.compute_accuracy())
            all_dim_reduc_time.append(dim_reduc_time)
            all_train_time.append(train_time)
            all_trans_stats.append(data_stats(reduc))

        return all_accuracy, all_dim_reduc_time, all_train_time, all_trans_stats

    def performance_test_all(self, data: performance_test_data, model: performance_test_model, dim_reduc_functions: List[dim_reduc_function]):
        original_data_stats = data_stats(data.training_data)
        result = []
        for func in dim_reduc_functions:
            all_accuracy, all_dim_reduc_time, all_train_time, all_trans_stats = self._performance_test_one_func(data, model, func)
            four_ways = zip(all_accuracy, all_dim_reduc_time, all_train_time, all_trans_stats)
            for acc, reduc_time, train_time, trans_stats in four_ways:
                stat = stats(
                    func.name,
                    func.params,
                    reduc_time,
                    train_time,
                    acc,
                    original_data_stats,
                    trans_stats,
                    data.characterstics
                )
                result.append(stat)
        return result

