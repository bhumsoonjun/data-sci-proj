import numpy as np
from jlt import jlt
from fjlt import fjlt
from utils import utils
from sklearn.decomposition import PCA
from performance_categorizer import performance_cat
from jlt.jlt import *

k = 100
n = 10000
d = 100000
a = -1000
b = 1000
std = 550
ep = 0.1
de = 0.1
n_test_per_clus = 10
num_test = 1

tester = performance_cat(n, d, a, b, std, k, n_test_per_clus, num_test)

reduc_k = int(24/ep**2 * np.log(1/de))

n_components = 2100
svd_solver = "auto"
model = PCA(n_components=n_components, svd_solver=svd_solver)

funcs = [lambda x: ese_transform(x, ep, de), lambda x: jlt_r(x, reduc_k), lambda x: jlt(x, reduc_k), lambda x: model.fit_transform(x)]
names = ["ese", "ran", "nor", "pca"]
params = [{"ep": ep, "de": de}, {"ep": ep, "de": de}, {"ep": ep, "de": de}, {"n_components": n_components, "svd_solver": svd_solver}]
result = tester.performance_test_all(names, funcs, params)

for res, name in zip(result, names):
    print(name, res)