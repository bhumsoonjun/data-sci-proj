import numpy as np
import numpy as np
from scipy.sparse import csr_matrix
import numpy as np
from scipy import sparse
import numpy.random as npr
import math
from scipy.linalg import null_space

def jlt(data: np.ndarray, ep: float, de: float) -> np.ndarray:
    n, d = data.shape
    k = int(24 / ep ** 2 * np.log(1 / de))
    np.random.choice([1, 0, -1], p=[1/6, 2/3, 1/6], size=(d, k))
    proj_mat = 1/np.sqrt(k) * np.random.normal(size=(d, k))
    return data @ proj_mat

def jlt_r(data: np.ndarray, ep: float, de: float) -> np.ndarray:
    n, d = data.shape
    k = int(24 / ep ** 2 * np.log(1 / de))
    density = 1/np.sqrt(k)
    s = 1/density
    proj_mat = np.sqrt(s/k) * np.random.choice([1, 0, -1], p=[1/(2*s),1 - 1/s, 1/(2*s)], size=(d, k))
    return data @ proj_mat

def jlt_ese(X, de: float, ep: float):
    n, d = X.shape
    k = int((2 * np.log(n) - np.log(de)) * np.log(d) / ep)
    h = np.random.choice(d, size=k, replace=True)
    sigma = np.random.choice([-1, 1], size=d)
    R = csr_matrix((sigma[h], (h, range(k))), shape=(d, k))
    return np.sqrt(d/k) * X @ R