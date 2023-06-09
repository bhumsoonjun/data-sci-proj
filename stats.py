from dataclasses import dataclass
import numpy as np
from typing import *
from data_stats import *
@dataclass(init=True, repr=True)
class stats:
    dim_reduc_method_name: str
    dim_reduc_params: Dict[str, Any]
    dim_reduc_time: float
    train_time: float
    accuracy: float

    original_data_stats: data_stats
    transformed_data_stats: data_stats

    characteristics: dict