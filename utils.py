import numpy as np


def measure_error(original: np.ndarray, processed: np.ndarray):
    n, m = original.shape
    gram_mat_ori = original @ original.T
    g_ori = np.diag(np.diag(gram_mat_ori)) # n x n
    D_ori = g_ori @ np.ones(shape=(n,1)) + np.ones(shape=(1, n)) @ g_ori - 2*g_ori

    n, m = processed.shape
    gram_mat_pro = processed @ processed.T
    g_pro = np.diag(np.diag(gram_mat_pro)) # m x m
    D_pro = g_pro @ np.ones(shape=(n,1)) + np.ones(shape=(1, n)) @ g_pro - 2*g_pro

    return np.abs(np.sum(np.sqrt(D_ori)) - np.sum(np.sqrt(D_pro)))
