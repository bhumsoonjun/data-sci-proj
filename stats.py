from dataclasses import dataclass
import numpy as np
from typing import *

@dataclass(init=True, repr=True)
class stats:
    dim_reduc_method_name: str
    dim_reduc_params: Dict[str, Any]
    dim_reduc_time: float
    train_time: float
    accuracy: float

    ori_stds_sum: float
    ori_stds_mean: float
    ori_shape: Tuple[int, int]

    trans_stds_sum: float
    trans_stds_mean: float
    trans_shape: Tuple[int, int]

    num_clusters: int
    a: int
    b: int
    cluster_std: float