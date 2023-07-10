from . import data_stats
from dataclasses import dataclass
from typing import *
@dataclass(init=True, repr=True)
class test_result:
    run_number: int

    dim_reduc_method_name: str
    dim_reduc_params: Dict[str, Any]
    dim_reduc_time: float
    train_time: float
    accuracy: float

    original_data_stats: data_stats
    transformed_data_stats: data_stats

    characteristics: dict