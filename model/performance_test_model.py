from abc import ABC, abstractmethod

from data_generator.performance_test_data import *

"""
Abstraction of model that we want to test. For example, kmeans, regression, and so on.
Necessary methods are defined in here to be override. 
This is for the sake of scalability of the code and my sanity as well.
"""

class performance_test_model(ABC):

    @abstractmethod
    def initialize(self, data: performance_test_data):
        pass

    @abstractmethod
    def apply_dim_reduc(self, dim_reduc_f) -> np.ndarray:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def compute_accuracy(self):
        pass
