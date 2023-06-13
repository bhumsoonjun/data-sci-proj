from typing import *
import numpy as np
from performance_test_data import *
from sklearn.model_selection import train_test_split

class linear_data_generator:

    def __init__(
        self,
        n: int,
        d: int,
        x_range: float,
        coeff_range: float,
        std: int,
        sparsity: float
    ):
        self.n = n
        self.d = d
        self.x_range = x_range
        self.coeff_range =coeff_range
        self.std = std
        self.sparsity = sparsity
        self.characteristics = {
            "n": n,
            "d": d,
            "x_range": x_range,
            "coeff_range": coeff_range,
            "std": std,
            "sparsity": sparsity
        }

    def _create_one_sparse_mask(self) -> np.ndarray:
        return np.random.binomial(1, np.array([1 - self.sparsity]).repeat(repeats=self.d - 1))

    def _create_sparse_mask_family(self):
        amount = int(1/(1 - self.sparsity) + 0.5)
        masks = np.array(list(map(lambda _: self._create_one_sparse_mask(), [i for i in range(amount)])))
        return masks

    def _make_sparse(self, data: np.ndarray, masks: np.ndarray) -> np.ndarray:
        chosen = np.random.randint(low=0, high=masks.shape[0], size=(data.shape[0],))
        return data * masks[chosen, :]

    def func(self, x: np.ndarray, w: np.ndarray, b: float):
        return x @ w + b

    def _project_onto(self, x: np.ndarray):
        amount = int(0.05 * x.shape[1])
        possible = np.arange(self.d - 1)
        index = np.random.choice(possible, size=(amount,), replace=False)
        cov_col = np.random.choice(np.setdiff1d(possible, index), size=(amount,), replace=False)
        masks = np.random.uniform(low=-np.sqrt(self.coeff_range), high=np.sqrt(self.coeff_range), size=(amount,))
        x[:, index] = x[:, cov_col] + x[:, index] * masks
        return x


    def _make_related(self, x: np.ndarray):
        amount = int(0.05 * x.shape[1])
        possible = np.arange(self.d - 1)
        index = np.random.choice(possible, size=(amount,), replace=False)
        cov_col = np.random.choice(np.setdiff1d(possible, index), size=(amount,), replace=False)
        masks = np.random.uniform(low=-self.coeff_range, high=self.coeff_range, size=(amount,))
        x[:, index] = x[:, cov_col] * masks
        return x


    def _create_test_data(self):
        mask_family = self._create_sparse_mask_family()
        random_coeffs = np.random.uniform(low=-self.coeff_range, high=self.coeff_range, size=(self.d, ))

        w = random_coeffs[:-1].reshape(-1, 1)
        b = random_coeffs[-1]
        x = self._make_sparse(np.random.uniform(low=-self.x_range, high=self.x_range, size=(self.n, self.d - 1)), mask_family)
        x = self._project_onto(self._make_related(x))
        y = self.func(x, w, b) + np.random.normal(scale=self.std, size=(self.n, 1))

        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2)
        return performance_test_data(training_data=x_train, training_label=y_train, testing_data=x_test, testing_label=y_test)

    def generate(self) -> performance_test_data:
        return self._create_test_data()