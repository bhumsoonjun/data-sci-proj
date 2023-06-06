import numpy as np
from jlt import jlt
from fjlt import fjlt
from utils import utils
from sklearn.decomposition import PCA
from performance_categorizer import performance_cat
from jlt.jlt import *

k = 10
n = 1000
d = 1000
a = -100
b = 100
std = 10000
ep = 0.1
de = 0.1
n_test_per_clus = 10
num_test = 1

tester = performance_cat(n, d, a, b, std, k, n_test_per_clus, num_test)

reduc_k = int(24/ep**2 * np.log(1/de))

model = PCA(n_components=100, svd_solver="full")
funcs = [lambda x: ese_transform(x, ep, de), lambda x: jlt_r(x, reduc_k), lambda x: jlt(x, reduc_k), lambda x: model.fit_transform(x)]
names = ["ese", "ran", "nor", "pca"]
result = tester.performance_test_all(funcs)

for res, name in zip(result, names):
    print(name, res)