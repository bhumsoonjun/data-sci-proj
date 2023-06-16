import time

import scipy as sc

from data_stats import *
from dim_reduc_function import *
from performance_test_model import *
from test_result import test_result


class performance_cat:

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
            dim_reduc_f: dim_reduc_function,
            num_test: int
    ):
        all_accuracy = []
        all_dim_reduc_time = []
        all_train_time = []
        all_trans_stats = []

        model.initialize(data)

        for i in range(num_test):
            print(f"==== Iter {i} ====")
            print(f"=== Applying Dimensionality Reduction: {dim_reduc_f.name} ===")
            print(f"=== Params: {dim_reduc_f.params} ===")
            reduc, dim_reduc_time = self._time_proc(lambda: model.apply_dim_reduc(dim_reduc_f.f))

            print("=== Traning Model ===")
            _, train_time = self._time_proc(lambda: model.train())

            print("=== Computing Accuracy ===")
            all_accuracy.append(model.compute_accuracy())
            all_dim_reduc_time.append(dim_reduc_time)
            all_train_time.append(train_time)
            all_trans_stats.append(data_stats(reduc))

        return all_accuracy, all_dim_reduc_time, all_train_time, all_trans_stats

    def performance_test_all(self, data: performance_test_data, model: performance_test_model, dim_reduc_functions: List[dim_reduc_function], num_test_funcs: List[int]):
        original_data_stats = data_stats(data.training_data)
        result = []
        print(dim_reduc_functions)
        for func, num_test in zip(dim_reduc_functions, num_test_funcs):
            all_accuracy, all_dim_reduc_time, all_train_time, all_trans_stats = self._performance_test_one_func(data, model, func, num_test)
            five_ways = zip(all_accuracy, all_dim_reduc_time, all_train_time, all_trans_stats, [i for i in range(num_test)])
            for acc, reduc_time, train_time, trans_stats, i in five_ways:
                stat = test_result(
                    i,
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

