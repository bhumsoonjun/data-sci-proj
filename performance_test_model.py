from abc import ABC, abstractmethod

from performance_test_data import *


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
