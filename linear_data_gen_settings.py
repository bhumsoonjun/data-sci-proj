from typing import *

class linear_data_gen_settings:

    def __init__(self, n: int, d: int, x_range: int, coeff_range: int, std: int, sparsity: float):
        self.n = n
        self.d = d
        self.x_range = x_range
        self.coeff_range = coeff_range
        self.std = std
        self.sparsity = sparsity
