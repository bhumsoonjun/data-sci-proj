from utils import utils
import numpy as np

def test_measure_error():
    arr_in = np.array([[1, 2, 3], [4, 5, 6]])
    arr_out = np.array([[1, 2], [4, 5]])
    error = utils.measure_error(arr_in, arr_out)
    print("\n==================")
    print(f"error: {error}")
