import numpy as np
from typing import *

def measure_error(original: np.ndarray, processed: np.ndarray):
    m, n = original.shape
    error = 0
    for i in range(m):
        for j in range(i + 1, m):
            transformed_diff = np.linalg.norm(processed[i, :] - processed[j, :])
            original_diff = np.linalg.norm(original[i, :] - original[j, :])
            error += np.abs(transformed_diff - original_diff)
    return error

