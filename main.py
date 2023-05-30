import numpy as np
from jlt import jlt
from fjlt import fjlt
from utils import utils
from ese_jlt import ese_jlt
from sklearn.decomposition import PCA

"""
A = np.random.randn(3, 5) * 100
res = fjlt(A, 1, 0.5)

print(res.shape)

mean_along_row = A.mean(axis = 1).reshape((3, 1))
F = A - mean_along_row
FT = F.T

print(mean_along_row)
print(F)
print(F @ FT)
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