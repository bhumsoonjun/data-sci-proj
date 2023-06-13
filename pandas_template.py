import pandas as pd
from test_result import *

structure = {
    "name": [],
    "run_num": [],
    "instance_num": [],
    "params": [],
    "reduction_time": [],
    "accuracy": [],
    "original_std_sum": [],
    "original_std_mean": [],
    "original_std_median": [],
    "original_std_max": [],
    "original_std_min": [],
    "original_sparsity": [],
    "transformed_std_sum": [],
    "transformed_std_mean": [],
    "transformed_std_median": [],
    "transformed_std_max": [],
    "transformed_std_min": [],
    "transformed_sparsity": [],
    "characteristics": []
}

def get_blank_dataframe():
    return pd.DataFrame(structure)

def inject_into_dataframe(all_res: List[test_result]):
    name = [res.dim_reduc_method_name for res in all_res]
    run_num = [res.run_number for res in all_res]
    instance_num = [res.instance_num for res in all_res]
    params = [res.dim_reduc_params for res in all_res]
    reduction_time = [res.dim_reduc_time for res in all_res]
    accuracy = [res.accuracy for res in all_res]
    original_std_sum = [res.original_data_stats.stds_sum for res in all_res]
    original_std_mean = [res.original_data_stats.stds_mean for res in all_res]
    original_std_median = [res.original_data_stats.stds_median for res in all_res]
    original_std_max = [res.original_data_stats.std_max for res in all_res]
    original_std_min = [res.original_data_stats.std_min for res in all_res]
    original_sparsity = [res.original_data_stats.sparsity for res in all_res]
    transformed_std_sum = [res.transformed_data_stats.stds_sum for res in all_res]
    transformed_std_mean = [res.transformed_data_stats.stds_mean for res in all_res]
    transformed_std_median = [res.transformed_data_stats.stds_median for res in all_res]
    transformed_std_max = [res.transformed_data_stats.std_max for res in all_res]
    transformed_std_min = [res.transformed_data_stats.std_min for res in all_res]
    transformed_sparsity = [res.transformed_data_stats.sparsity for res in all_res]
    characteristics = [res.characteristics for res in all_res]

    structure_copy = structure.copy()

    structure_copy["name"] = name
    structure_copy["run_num"] = run_num
    structure_copy["instance_num"] = instance_num
    structure_copy["params"] = params
    structure_copy["reduction_time"] = reduction_time
    structure_copy["accuracy"] = accuracy
    structure_copy["original_std_sum"] = original_std_sum
    structure_copy["original_std_mean"] = original_std_mean
    structure_copy["original_std_median"] = original_std_median
    structure_copy["original_std_max"] = original_std_max
    structure_copy["original_std_min"] = original_std_min
    structure_copy["original_sparsity"] = original_sparsity
    structure_copy["transformed_std_sum"] = transformed_std_sum
    structure_copy["transformed_std_mean"] = transformed_std_mean
    structure_copy["transformed_std_median"] = transformed_std_median
    structure_copy["transformed_std_max"] = transformed_std_max
    structure_copy["transformed_std_min"] = transformed_std_min
    structure_copy["transformed_sparsity"] = transformed_sparsity
    structure_copy["characteristics"] = characteristics

    return pd.DataFrame(structure_copy)
