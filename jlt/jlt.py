import numpy as np
from typing import *
from scipy import sparse

def jlt(data: np.ndarray, k: int) -> np.ndarray:
    n, d = data.shape
    np.random.choice([1, 0, -1], p=[1/6, 2/3, 1/6], size=(d, k))
    proj_mat = 1/np.sqrt(k) * np.random.normal(size=(d, k))
    return data @ proj_mat

def jlt_r(data: np.ndarray, k: int) -> np.ndarray:
    n, d = data.shape
    density = 1/np.sqrt(k)
    s = 1/density
    print("=== sampling ===")
    proj_mat = np.sqrt(s/k) * np.random.choice([1, 0, -1], p=[1/(2*s),1 - 1/s, 1/(2*s)], size=(d, k))
    print("=== mult ===")
    return data @ proj_mat