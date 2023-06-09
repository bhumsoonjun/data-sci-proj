from typing import *
import numpy as np
from dataclasses import *


@dataclass(repr=True)
class data_stats:

    stds_sum: float
    stds_mean: float
    shape: Any
    sparsity: float

    def __init__(self, data: np.ndarray):
        self.stds = data.std(axis=1)
        self.stds_sum = self.stds.sum()
        self.stds_mean = self.stds.mean()
        self.shape = data.shape
        self.sparsity = 1.0 - np.count_nonzero(data) / float(data.size)
