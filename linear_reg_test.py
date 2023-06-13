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
from pandas_template import *

""" DATA GENERATION SETTINGS """

n = 100
d = 1000
x_range = 10000
coeff_range = 100
std = 10
sparsity = 0.99
num_test_total = 10

""" DIM REDUC SETTINGS """

ep = 0.9
de = 0.1
instance_num = 2
n_components = 99

svd_solver = "auto"
model = PCA(n_components=n_components, svd_solver=svd_solver)

""" General Settings """

output_path = f"output/lin_reg_{ep}_{de}_{instance_num}"

""" FUNCS """

ese_jlt = dim_reduc_function("extremely sparse JL transform", lambda x: jlt_ese(x, ep, de), {"ep": ep, "de": de}, instance_num)
random_jlt = dim_reduc_function("sparse JL transform", lambda x: jlt_r(x, ep, de), {"ep": ep, "de": de}, instance_num)
n_jlt = dim_reduc_function("JL transform", lambda x: jlt(x, ep, de), {"ep": ep, "de": de}, instance_num)
pca = dim_reduc_function("PCA", lambda x: model.fit_transform(x),  {"n_components": n_components, "svd_solver": svd_solver}, instance_num)
blank = dim_reduc_function("Nothing", lambda x: x, {}, instance_num)
funcs = [ese_jlt, random_jlt, n_jlt, pca, blank]
num_test_funcs = [10, 10, 10, 10, 1]

for i in range(num_test_total):
    reg = regression_model()
    gen = linear_data_generator(n, d, x_range, coeff_range, std, sparsity)
    data = gen.generate()
    tester = performance_cat()
    results = tester.performance_test_all(data, reg, funcs, num_test_funcs)
    dataframe = inject_into_dataframe(results)
    dataframe.to_csv(output_path, index=False, mode='a')
