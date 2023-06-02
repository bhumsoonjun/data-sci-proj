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
    labels = np.empty(shape=(1, 1), dtype=np.intc)
    for i in range(m):
        rand = np.random.normal(loc=0, scale=std, size=(n_per_cluster, d)) + clusters_mean[i, :]
        clusters_test = np.concatenate((clusters_test, rand))
        labels = np.concatenate((labels, np.full(shape=(n_per_cluster, 1), fill_value=i)))
    return clusters_test[1:, :], labels[1:, :].flatten()

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



def compute_accuracy(predicted: np.ndarray, all_clusters: np.ndarray, labels: np.ndarray):
    n, d = predicted.shape
    closest_all = np.empty(shape=(n,))
    for i in range(n):
        dir_vecs = predicted[i, :] - all_clusters
        dist_vecs = np.sum(dir_vecs * dir_vecs, axis=1)
        closest_all[i] = np.argmin(dist_vecs)
    closest_all = closest_all.flatten()
    return np.sum(closest_all == labels)

def performance_test(n: int, a: int, b: int, std: int, k: int, dim_reduc_f):
    print("===== Creating Training Cluster =====")
    clusters_train, clusters_means = create_cluster(k, n, d, a, b, std)
    print("===== Creating Test Cluster =====")
    clusters_test, labels = create_test_cluster(clusters_means, n//k, std)
    print("===== Testing =====")
    all_accuracy = []
    all_dim_reduc_time = []
    all_train_time = []
    all_dim = []
    for i in range(3):
        print(f"==== Iter {i} ====")
        print("=== Applying Dimensionality Reduction ===")
        reduc, dim_reduc_time = time_proc(lambda: apply_dim_reduc(dim_reduc_f, np.concatenate((clusters_train, clusters_test, clusters_means))))
        reduc_train, reduc_test, reduc_means = reduc[:n, :], reduc[n:n+clusters_test.shape[0], :], reduc[n+clusters_test.shape[0]:, :]
        print("=== Traning Model ===")
        model, train_time = time_proc(lambda: KMeans(n_clusters=k).fit(reduc_train))
        print("=== Predicting Model ===")
        predicted = model.predict(reduc_test)
        print("=== Computing Accuracy ===")
        all_accuracy.append(compute_accuracy(model.cluster_centers_[predicted], reduc_means, labels))
        all_dim_reduc_time.append(dim_reduc_time)
        all_train_time.append(train_time)
        all_dim.append(reduc.shape[1])
    return all_accuracy, all_dim_reduc_time, all_train_time, all_dim

k = 100
n = 10000
d = 10000
a = -10
b = 10
std = 10
ep = 0.5
de = 0.5

res = performance_test(n, a, b, std, k, lambda x: ese_transform(x, ep, de))
print(res)
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
