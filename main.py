from sklearn.decomposition import PCA

from clusters_generator import *
from dim_reduc_function import *
from jlt import jlt
from jlt.jlt import *
from kmeans_model import *
from performance_categorizer import *
import pandas as pd
from scipy.io.arff import loadarff

n = 1000
d = 10000
a = -100
b = 100
cluster_std = 1000
num_cluster = 10
ep = 0.1
de = 0.1
n_test_per_clus = 10
num_test = 1
sparsity = 0.9

reduc_k = int(24/ep**2 * np.log(1/de))

n_components = 1000
svd_solver = "auto"
model = PCA(n_components=n_components, svd_solver=svd_solver)

""" FUNCS """

ese_jlt = dim_reduc_function("extremely sprase JL transform", lambda x: jlt_ese(x, ep, de), {"ep": ep, "de": de})
random_jlt = dim_reduc_function("sparse JL transform", lambda x: jlt_r(x, reduc_k), {"ep": ep, "de": de})
n_jlt = dim_reduc_function("JL transform", lambda x: jlt(x, reduc_k), {"ep": ep, "de": de})
pca = dim_reduc_function("PCA", lambda x: model.fit_transform(x),  {"n_components": n_components, "svd_solver": svd_solver})

funcs = [ese_jlt, random_jlt, n_jlt, pca]

""" MODEL """

kmeans = kmeans_model()

""" DATA """
cg = clusters_generator(n, d, a, b, cluster_std, num_cluster, n_test_per_clus, sparsity)
performance_test_data = cg.generate()

tester = performance_cat(num_test)
results = tester.performance_test_all(performance_test_data, kmeans, funcs)

for res in results:
    print(res)
