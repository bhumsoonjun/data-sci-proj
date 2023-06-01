import numpy as np
import sklearn as sk
import scipy as sc
from typing import *
import time
from sklearn.model_selection import train_test_split

def create_one_cluster(n_par: int, d: int, a: int, b: int, std:float = 1):
    rand_mean = np.random.uniform(high=b, low=a, size=(1, d))
    return np.random.normal(loc=0, scale=std, size=(n_par, d)) + rand_mean, rand_mean

def create_cluster(k: int, n: int, d: int,  a: int, b:int, std: float = 1):
    clusters = np.empty(shape=(1, d))
    means = np.empty(shape=(1, d))
    for i in range(k):
        cluster, mean = create_one_cluster(n//k, d, a, b, std)
        means = np.concatenate((means, mean))
        clusters = np.concatenate((clusters, cluster))
    return clusters[1:, :], means[1:, :]

def create_test_cluster(clusters_mean: np.ndarray, n_per_cluster: int, std):
    m, d = clusters_mean.shape
    clusters_test = np.empty(shape=(1, d))
    for i in range(m):
        rand = np.random.normal(loc=0, scale=std, size=(n_per_cluster, d)) + clusters_mean[i, :]
        clusters_test = np.concatenate((clusters_test, rand))
    return clusters_test[1:, :]

def rescale(data: np.ndarray):
    stds = data.std(axis=0)
    return sc.cluster.vq.whiten(clusters), stds

def reverse_rescale(data: np.ndarray, stds: np.ndarray):
    return data * stds

def apply_dim_reduc(f, data):
    return f(data)

def time_proc(f):
    start = time.perf_counter_ns()
    res = f()
    return res, time.perf_counter_ns() - start

def performance_test(n: int, a: int, b: int, std: int, k: int, dim_reduc_f):
    clusters_train, clusters_means = create_cluster(k, n, d, a, b, std)
    clusters_test = create_test_cluster(clusters_means, 100, std)
    for i in range(10):
        reduc, dim_reduc_time = time_proc(lambda: apply_dim_reduc(dim_reduc_f, clusters_train))
        scaled, stds = rescale(reduc)
        k_clusters, _ = time_proc(lambda: sc.cluster.vq.kmeans(scaled, k))
        reversed = reverse_rescale(k_clusters, stds)

k = 2
n = 10000
d = 1000
a = -1000
b = 1000
std = 1000
clusters, means = create_cluster(k, n, d, a, b, std)
print(clusters.shape)

whitened, stds = rescale(clusters)
k_clusters, _ = sc.cluster.vq.kmeans(whitened, k)

print(stds)
print(k_clusters)
print(k_clusters * stds)
print(means)
