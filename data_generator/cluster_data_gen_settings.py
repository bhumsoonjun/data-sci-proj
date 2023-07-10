from dataclasses import dataclass

"""
A dataclass for holding data generation setting for cluster data type.
"""
@dataclass(repr=True)
class cluster_data_gen_settings:
    n: int
    d: int
    a: int
    b: int
    std: float
    num_clusters: int
    num_test_per_cluster: int
    sparsity: float

    def __init__(self, n: int, d: int, a: int, b: int, std: float, num_clusters: int, num_test_per_cluster: int, sparsity: float):
        self.n = n
        self.d = d
        self.a = a
        self.b = b
        self.std = std
        self.num_clusters = num_clusters
        self.num_test_per_cluster = num_test_per_cluster
        self.sparsity = sparsity
