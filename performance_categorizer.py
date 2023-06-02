import numpy as np
import sklearn as sk
import scipy as sc
from typing import *
import time
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from jlt.jlt import *

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
    labels = np.empty(shape=(1, 1))
    for i in range(m):
        rand = np.random.normal(loc=0, scale=std, size=(n_per_cluster, d)) + clusters_mean[i, :]
        clusters_test = np.concatenate((clusters_test, rand))
        print(labels.shape, np.full(shape=(n_per_cluster, 1), fill_value=i).shape)
        labels = np.concatenate((labels, np.full(shape=(n_per_cluster, 1), fill_value=i)))
    return clusters_test[1:, :], labels[1:, :]

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

def compute_accuracy(predicted: np.ndarray, label: np.ndarray, all_clusters: np.ndarray):
    print(predicted.shape)
    n, _ = predicted.reshape(-1, 1).shape
    closest_all = np.empty(shape=(1, 1))
    print(predicted.shape)
    for i in range(n):
        print(predicted[i])
        print(all_clusters)
        dist_vec = predicted[i] - all_clusters
        data_i = np.dot(dist_vec, dist_vec)
        closest_all = np.concatenate((closest_all, np.argmin(data_i)))
    return np.sum(closest_all[1] == label)

def performance_test(n: int, a: int, b: int, std: int, k: int, dim_reduc_f):
    clusters_train, clusters_means = create_cluster(k, n, d, a, b, std)
    clusters_test, labels = create_test_cluster(clusters_means, 100, std)
    all_accuracy = []
    print("===== now =====")
    for i in range(1):
        reduc, dim_reduc_time = time_proc(lambda: apply_dim_reduc(dim_reduc_f, np.concatenate((clusters_train, clusters_test))))
        reduc_train, reduc_test = reduc[:n, :], reduc[n:, :]
        model, _ = time_proc(lambda: KMeans(n_clusters=k).fit(reduc_train))
        predicted = model.predict(reduc_test)
        score = compute_accuracy(predicted, actual)
    return all_accuracy

k = 2
n = 10000
d = 10000
a = -1000
b = 1000
std = 10
ep = 0.01
de = 0.01

performance_test(n, a, b, std, k, lambda x: ese_transform(x, ep, de))

"""
clusters, means = create_cluster(k, n, d, a, b, std)
print(clusters.shape)

whitened, stds = rescale(clusters)
k_clusters, _ = sc.cluster.vq.kmeans(whitened, k)

print(stds)
print(k_clusters)
print(k_clusters * stds)
print(means)
"""
