import numpy as np
from scipy.sparse import csr_matrix

def ese_transform(X, delta, epsilon):
    """
    Perform Extremely Sparse Johnson-Lindenstrauss Transform (ESE) on the input data.

    Parameters:
    - X: Input data matrix of shape (n, d), where n is the number of samples and d is the dimensionality.
    - delta: Desired probability guarantee (0 < delta < 1).
    - epsilon: Desired approximation factor (0 < epsilon < 1).

    Returns:
    - X_hat: Transformed data matrix of shape (n, k), where k is the transformed dimensionality.
    """
    n, d = X.shape

    # Calculate the target dimensionality k
    k = int((2 * np.log(n) - np.log(delta)) * np.log(d) / epsilon)
    print(k)
    # Build hash function h
    h = np.random.choice(d, size=k, replace=True)
    print(h)
    # Build hash function sigma
    sigma = np.random.choice([-1, 1], size=d)
    print(sigma)
    # Build matrix R
    R = csr_matrix((sigma[h], (h, range(k))), shape=(d, k))

    print(f"Non zeros: {R.count_nonzero()}")
    # Compute X_hat
    X_hat = np.sqrt(d/k) * X @ R

    return X_hat