import numpy as np

from template.dim_reduc_function import *
from sklearn.decomposition import PCA
from dataclasses import dataclass

"""
A wrapper class for PCA. This is to make it conform with other dimensionality reduction algorithm.
Also goes with adapter design pattern to wrap another functionality and convert it to similar signature for OOP purposes.
"""
@dataclass(repr=True)
class pca_wrapper(dim_reduc_function):

    def __init__(self, name, f, params: dict):
        self.name: str = name
        self.f = self.apply
        self.params = params

    def apply(self, data: np.ndarray):
        n_com = int(data.shape[0]//100)
        model = PCA(n_components=n_com, svd_solver="auto")
        self.params["n_components"] = n_com
        self.params["svd_solver"] = "auto"
        return model.fit_transform(data)