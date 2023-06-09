from typing import *
import numpy as np

class performance_test_data:

    def __init__(self, training_data, training_label, testing_data, testing_label, characteristics: dict = None):
        self.training_data: np.ndarray = training_data
        self.testing_data: np.ndarray = testing_data
        self.testing_label: np.ndarray = testing_label
        self.training_label: np.ndarray = training_label
        self.characterstics: dict = characteristics

