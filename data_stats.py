from typing import *
import numpy as np

class data_stats:

    def __init__(self, data: np.ndarray):
        self.stds = data.std(axis=1)
        self.stds_sum = self.stds.sum()
        self.stds_mean = self.stds.mean()
        self.shape = data.shape
        self.sparsity = 1.0 - np.count_nonzero(data) / float(data.size)