import sklearn as sk
import sklearn.cluster

from performance_test_model import *
import sklearn as sk
import sklearn.cluster

from performance_test_model import *


class regression_model(performance_test_model):

    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

        self.n = None
        self.d = None
        self.model: sklearn.linear_model.LinearRegression = None

        self.reduc_train_data = None
        self.reduc_test_data = None

    def initialize(self, data: performance_test_data):
        self.train_data = data.training_data
        self.train_label = data.training_label
        self.test_data = data.testing_data
        self.test_label = data.testing_label
        self.n, self.d = self.train_data.shape
        self.model: sklearn.linear_model.LinearRegression = sk.linear_model.LinearRegression()

        self.compact_data = np.concatenate((self.train_data, self.test_data))


    def apply_dim_reduc(self, dim_reduc_f) -> np.ndarray:
        reduc = dim_reduc_f(self.compact_data)
        self.reduc_train_data = reduc[:self.n, :]
        self.reduc_test_data = reduc[self.n:, :]
        return reduc

    def train(self):
        self.model.fit(self.reduc_train_data, self.train_label)

    def compute_accuracy(self):
        return self.model.score(self.reduc_test_data, self.test_label)
