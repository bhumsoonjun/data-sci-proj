from health_news_parser import *
from jlt import jlt
from jlt.jlt import *
from linear_data_gen_settings import *
from linear_data_generator import *
from pandas_template import *
from performance_categorizer import *
from regression_model import *
from sklearn.decomposition import PCA
import torch
print(torch.cuda.is_available())

""" TEST SETTINGS """

num_test_total = 3
num_test_each = 3

""" DATA GENERATION SETTINGS """

std_settings = [10, 100, 1000]
sparsity_settings = [0, 0.3, 0.5, 0.7, 0.99]
n_settings = [1000, 10000, 100000]
d_settings = [1000, 10000, 100000]
settings = [
    (linear_data_gen_settings(n=n, d=d, x_range=10000, coeff_range=100, std=i, sparsity=j), f"output/lin_reg/{i}_{j}")
    for i in std_settings
    for j in sparsity_settings
    for n in n_settings
    for d in d_settings
]

""" DIM REDUC SETTINGS """
eps = [0.05, 0.1, 0.3, 0.5, 0.9]
des = [0.05, 0.1, 0.3]
n_components_arr = [100, 1000, 2000]
instance_nums = [j + (i * j) for i in range(len(eps)) for j in range(len(des))]
ep = 0.9
de = 0.1
instance_num = 2
n_components = 99

svd_solver = "randomized"
model = PCA(n_components=n_components, svd_solver=svd_solver)

""" FUNCS """

ese_jlt = [dim_reduc_function("extremely sparse JL transform", lambda x: jlt_ese(x, ep, de), {"ep": ep, "de": de}) for ep in eps for de in des]
random_jlt = [dim_reduc_function("sparse JL transform", lambda x: jlt_r(x, ep, de), {"ep": ep, "de": de}) for ep in eps for de in des]
n_jlt = [dim_reduc_function("JL transform", lambda x: jlt(x, ep, de), {"ep": ep, "de": de}) for ep in eps for de in des]
pca = [dim_reduc_function("PCA", lambda x: model.fit_transform(x),  {"n_components": n_components, "svd_solver": svd_solver}) for n_components in n_components_arr]
blank = [dim_reduc_function("Nothing", lambda x: x, {})]
funcs = ese_jlt + random_jlt + n_jlt + pca + blank
num_test_funcs = [num_test_each for i in range(len(eps) * len(des) * 3)] + [3 for i in range(len(n_components_arr))] + [3]

for setting in settings:
    print(f"========= Settings = {setting} ==========")
    lin_setting, output_path = setting
    n, d, x_range, coeff_range, std, sparsity = lin_setting.n, lin_setting.d, lin_setting.x_range, lin_setting.coeff_range, lin_setting.std, lin_setting.sparsity
    for i in range(num_test_total):
        print(f"========= Test No = {i} ==========")
        reg = regression_model()
        gen = linear_data_generator(n, d, x_range, coeff_range, std, sparsity)
        data = gen.generate()
        tester = performance_cat()
        results = tester.performance_test_all(data, reg, funcs, num_test_funcs)
        dataframe = inject_into_dataframe(results)
        dataframe.to_csv(output_path, index=False, mode='a')
