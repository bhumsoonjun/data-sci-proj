from dataclasses import *
from typing import *

import numpy as np

"""
Dataclass representation of statistics collected from data.
The init method takes in an array and collect statistics on the data.
"""
@dataclass(repr=True)
class data_stats:

    stds_sum: float
    stds_mean: float
    stds_median: float
    std_max: float
    std_min: float
    dist_from_origin: float
    shape: Any
    sparsity: float

    def __init__(self, data: np.ndarray):
        self.stds = data.std(axis=1)
        self.stds_sum = self.stds.sum()
        self.stds_mean = self.stds.mean()
        self.stds_median = np.median(self.stds)
        self.std_max = np.max(self.stds)
        self.std_min = np.min(self.stds)
        self.dist_from_origin = np.linalg.norm(data.mean(axis=0))
        self.shape = data.shape
        self.sparsity = 1.0 - np.count_nonzero(data) / float(data.size)
