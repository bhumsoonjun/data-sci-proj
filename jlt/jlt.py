import numpy as np
import numpy as np
from scipy.sparse import csr_matrix
import numpy as np
from scipy import sparse
import numpy.random as npr
import math
from scipy.linalg import null_space

from dim_reduc_function import dim_reduc_function


class jlt(dim_reduc_function):
    def __init__(self, name, ep: float, de: float, params: dict):
        super().__init__(name, None, params)
        self.ep = ep
        self.de = de

    def apply(self, data):
        n, d = data.shape
        k = int(1 / self.ep ** 2 * np.log(1 / self.de))
        proj_mat = 1/np.sqrt(k) * np.random.normal(size=(d, k))
        return data @ proj_mat

class jlt_r(dim_reduc_function):
    def __init__(self, name, ep: float, de: float, params: dict):
        super().__init__(name, None, params)
        self.ep = ep
        self.de = de

    def apply(self, data):
        n, d = data.shape
        k = int(1 / self.ep ** 2 * np.log(1 / self.de))
        density = 1/np.sqrt(k)
        s = 1/density
        proj_mat = np.sqrt(s/k) * np.random.choice([1, 0, -1], p=[1/(2*s),1 - 1/s, 1/(2*s)], size=(d, k))
        return data @ proj_mat

class jlt_ese(dim_reduc_function):

    def __init__(self, name, ep: float, de: float, params: dict):
        super().__init__(name, None, params)
        self.de = de
        self.ep = ep

    def apply(self, data):
        self.n, self.d = data.shape
        self.k = int((2 * np.log(self.n) - np.log(self.de)) * np.log(self.d) / self.ep)
        h = np.random.choice(self.d, size=self.k, replace=True)
        sigma = np.random.choice([-1, 1], size=self.d)
        R = csr_matrix((sigma[h], (h, range(self.k))), shape=(self.d, self.k))
        return np.sqrt(self.d/self.k) * data @ R