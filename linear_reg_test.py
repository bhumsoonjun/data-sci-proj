from sklearn.decomposition import PCA

from clusters_generator import *
from dim_reduc_function import *
from jlt import jlt
from jlt.jlt import *
from kmeans_model import *
from performance_categorizer import *
import pandas as pd
from scipy.io.arff import loadarff
from health_news_parser import *
from linear_data_generator import *
from regression_model import *

n = 100
d = 1000
x_range = 10000
coeff_range = 100
std = 10
sparsity = 0.99
num_test = 10

""" DIM REDUC SETTINGS """

ep = 0.3
de = 0.1
reduc_k = int(24/ep**2 * np.log(1/de))

print(reduc_k)

n_components = 99

svd_solver = "auto"
model = PCA(n_components=n_components, svd_solver=svd_solver)

""" FUNCS """

ese_jlt = dim_reduc_function("extremely sprase JL transform", lambda x: jlt_ese(x, ep, de), {"ep": ep, "de": de})
random_jlt = dim_reduc_function("sparse JL transform", lambda x: jlt_r(x, reduc_k), {"ep": ep, "de": de})
n_jlt = dim_reduc_function("JL transform", lambda x: jlt(x, reduc_k), {"ep": ep, "de": de})
pca = dim_reduc_function("PCA", lambda x: model.fit_transform(x),  {"n_components": n_components, "svd_solver": svd_solver})
blank = dim_reduc_function("Nothing", lambda x: x, {})
funcs = [ese_jlt, random_jlt, n_jlt, pca]

""" MODEL """

reg = regression_model()

""" DATA """
gen = linear_data_generator(n, d, x_range, coeff_range, std, sparsity)
data = gen.generate()

tester = performance_cat(num_test)
results = tester.performance_test_all(data, reg, funcs)

for res in results:
    print(res)
