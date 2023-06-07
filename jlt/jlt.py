import numpy as np
import numpy as np
from scipy.sparse import csr_matrix
import numpy as np
import fht  # Fast hadamard transform. https://github.com/nbarbey/fht
from scipy import sparse
import numpy.random as npr
import math

def jlt(data: np.ndarray, k: int) -> np.ndarray:
    n, d = data.shape
    np.random.choice([1, 0, -1], p=[1/6, 2/3, 1/6], size=(d, k))
    proj_mat = 1/np.sqrt(k) * np.random.normal(size=(d, k))
    return data @ proj_mat

def jlt_r(data: np.ndarray, k: int) -> np.ndarray:
    n, d = data.shape
    density = 1/np.sqrt(k)
    s = 1/density
    proj_mat = np.sqrt(s/k) * np.random.choice([1, 0, -1], p=[1/(2*s),1 - 1/s, 1/(2*s)], size=(d, k))
    return data @ proj_mat

def ese_transform(X, delta, epsilon):
    n, d = X.shape
    k = int((2 * np.log(n) - np.log(delta)) * np.log(d) / epsilon)
    h = np.random.choice(d, size=k, replace=True)
    sigma = np.random.choice([-1, 1], size=d)
    R = csr_matrix((sigma[h], (h, range(k))), shape=(d, k))

    return np.sqrt(d/k) * X @ R

def approx_bound(eps, n):
    return int(2 / eps ** 2 * math.log(n) + 1.0)


def fast_sample(n, sample_size):
    swap_records = {}
    sample_wor = np.empty(sample_size, dtype=int)
    for i in range(sample_size):
        rand_ix = npr.randint(i, n)

        if i in swap_records:
            el1 = swap_records[i]
        else:
            el1 = i

        if rand_ix in swap_records:
            el2 = swap_records[rand_ix]
        else:
            el2 = rand_ix

        swap_records[rand_ix] = el1
        sample_wor[i] = el2
        if i in swap_records:
            del swap_records[i]
    return sample_wor


def nextPow(d_act):
    d_act = d_act - 1
    d_act |= d_act >> 1
    d_act |= d_act >> 2
    d_act |= d_act >> 4
    d_act |= d_act >> 8
    d_act |= d_act >> 16
    d_act += 1
    return d_act


def fjlt(A, k, q):
    (d, n) = A.shape
    # Calculate the next power of 2
    d_act = nextPow(d)
    sc_ft = np.sqrt(d_act / float(d * k))
    # Calculate D plus some constansts
    D = npr.randint(0, 2, size=(d, 1)) * 2 * sc_ft - sc_ft
    DA = np.zeros((d_act, n))
    DA[0:d, :] = A * D

    # Apply hadamard transform to each row
    hda = np.apply_along_axis(fht.fht, 0, DA)

    P_ber = np.random.choice([1, 0], size=(k, d_act), p=[q, 1 - q])
    ind = np.where(P_ber == 1)
    num_samples = ind[0].shape[0]
    sample = np.random.normal(0, 1/q, num_samples)
    P = csr_matrix((sample, (ind[0], ind[1])), shape=(k, d_act))
    return P.dot(hda)


def fjlt_usp(A, k):
    (d, n) = A.shape
    # Calculate the next power of 2
    d_act = nextPow(d)
    sc_ft = np.sqrt(d_act / float(d * k))
    # Calculate D plus some constansts
    D = npr.randint(0, 2, size=(d, 1)) * 2 * sc_ft - sc_ft
    DA = np.zeros((d_act, n))
    DA[0:d, :] = A * D

    # Apply hadamard transform to each row
    hda = np.apply_along_axis(fht.fht, 0, DA)

    # Apply P transform
    p_cols = fast_sample(d, k)
    p_rows = np.array(range(k))
    p_data = npr.randint(0, 2, size=k) * 2 - 1
    P = sparse.csr_matrix((p_data, (p_rows, p_cols)), shape=(k, d_act))
    return P.dot(hda)