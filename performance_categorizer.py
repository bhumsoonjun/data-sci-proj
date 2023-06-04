import numpy as np
import sklearn as sk
import scipy as sc
from typing import *
import time

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from jlt.jlt import *

def create_one_cluster(n_par: int, d: int, a: int, b: int, std:float = 1):
    rand_mean = np.random.uniform(high=b, low=a, size=(1, d))
    return np.random.normal(loc=0, scale=std, size=(n_par, d)) + rand_mean, rand_mean

def create_one_cluster_with_mean(n_per_cluster: int, d: int, std:float, mean: np.ndarray):
    return np.random.normal(loc=0, scale=std, size=(n_per_cluster, d)) + mean

def create_cluster(k: int, n: int, d: int,  a: int, b:int, std: float = 1):
    means = np.random.uniform(high=b, low=a, size=(k, d))
    clusters = np.random.normal(loc=0, scale=std, size=(n, d)) + means.repeat(repeats=n//k, axis=0)
    return clusters, means

def create_test_cluster(clusters_mean: np.ndarray, n_per_cluster: int, std):
    n, d = clusters_mean.shape
    clusters_test = np.random.normal(loc=0, scale=std, size=(n * n_per_cluster, d)) + clusters_mean.repeat(repeats=n_per_cluster, axis=0)
    labels = np.arange(n).repeat(repeats=n_per_cluster)
    return clusters_test, labels.flatten()

def rescale(data: np.ndarray):
    stds = data.std(axis=0)
    return sc.cluster.vq.whiten(data), stds

def reverse_rescale(data: np.ndarray, stds: np.ndarray):
    return data * stds

def apply_dim_reduc(f, data):
    return f(data)

def time_proc(f):
    start = time.perf_counter_ns()
    res = f()
    return res, (time.perf_counter_ns() - start)/1e9

def find_min_indices(point: np.ndarray, all_clusters):
    dir = point - all_clusters
    dist = np.sum(dir * dir, axis=1)
    return np.argmin(dist)
def compute_accuracy(predicted: np.ndarray, predicted_actual: np.ndarray, n_test_per_cluster: int):
    return np.sum(predicted == predicted_actual.repeat(repeats=n_test_per_cluster)) / predicted.shape[0]

def create_test_data(n: int, a: int, b: int, std: int, k: int, n_test_per_cluster: int):
    print("===== Creating Training Cluster =====")
    clusters_train, clusters_means = create_cluster(k, n, d, a, b, std)
    print("===== Creating Test Cluster =====")
    clusters_test, labels = create_test_cluster(clusters_means, n_test_per_cluster, std)
    return clusters_train, clusters_means, clusters_test, labels

def performance_test_one_func(
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
    all_dim = []
    for i in range(num_test):
        print(f"==== Iter {i} ====")
        print("=== Applying Dimensionality Reduction ===")
        reduc, dim_reduc_time = time_proc(lambda: apply_dim_reduc(dim_reduc_f, np.concatenate((clusters_train, clusters_test, clusters_means))))
        reduc_train, reduc_test, reduc_means = reduc[:n, :], reduc[n:n+clusters_test.shape[0], :], reduc[n+clusters_test.shape[0]:, :]
        print("=== Traning Model ===")
        model, train_time = time_proc(lambda: KMeans(n_clusters=k).fit(reduc_train))
        print("=== Predicting Model ===")
        predicted = model.predict(reduc_test)
        predicted_actual = model.predict(reduc_means)
        print("=== Computing Accuracy ===")
        all_accuracy.append(compute_accuracy(predicted, predicted_actual, n_test_per_cluster))
        all_dim_reduc_time.append(dim_reduc_time)
        all_train_time.append(train_time)
        all_dim.append(reduc.shape[1])
    return all_accuracy, all_dim_reduc_time, all_train_time, all_dim

def performance_test_all(n: int, a: int, b: int, std: int, k: int, n_test_per_cluster: int, num_test: int, dim_reduc_f):
    clusters_train, clusters_means, clusters_test, labels = create_test_data(n, a, b, std, k, n_test_per_cluster)
    result = []
    for func in dim_reduc_f:
        res = performance_test_one_func(clusters_train, clusters_means, clusters_test, num_test, n_test_per_cluster, func)
        result.append(res)
    return result

k = 10
n = 1000
d = 100000
a = -100
b = 100
std = 10000
ep = 0.1
de = 0.1
n_test_per_clus = 1
num_test = 1

reduc_k = int(24/ep**2 * np.log(1/de))

model = PCA(n_components=100, svd_solver="auto")
funcs = [lambda x: ese_transform(x, ep, de), lambda x: jlt_r(x, reduc_k), lambda x: jlt(x, reduc_k), lambda x: model.fit_transform(x)]
names = ["ese", "ran", "nor", "pca"]
result = performance_test_all(n, a, b, std, k, n_test_per_clus, num_test, funcs)

for res, name in zip(result, names):
    print(name, res)
