import numpy as np
from jlt import jlt
from fjlt import fjlt
from utils import utils
from sklearn.decomposition import PCA


n = 100
d = 100000
ep = 0.05
delta = 0.05
A = np.random.randn(n, d)
# k = int(np.log(n)/ep**2) for fjlt
k = int(16/ep**2 * np.log(1/delta))

print("========== 1 ===========")

res = jlt.ese_transform(A, 0.2, 0.05)

print(res.shape)
print(utils.measure_error(A, res))

del res

print("=====================")

res = jlt.jlt_r(A, k)

print(res.shape)
print(utils.measure_error(A, res))

del res
"""

data = np.random.randn(1000, 100000)
print("========")
n, d = data.shape
ep = 0.3
k = int(24 / (ep ** 2) * np.log(n))
proj = ese_jlt.ese_transform(data, 0.05, 0.05)
print(k)
print(proj)
print(proj.shape)
# print(utils.measure_error(data, proj))

print("==== 2 ====")

proj = jlt.jlt_r(data, k)
#  print(utils.measure_error(data, proj))

print("==== 2 ====")

proj = jlt.jlt(data, k)
#  print(utils.measure_error(data, proj))

print("=================")
pca = PCA(n_components=10, svd_solver="full")
pca.fit(data.transpose())
print(pca.get_precision())
p = pca.transform(data.transpose())
print(pca.explained_variance_ratio_)
print(int(np.log(n)))
print(pca.components_.shape)

"""