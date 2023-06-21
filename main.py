from clusters_generator import clusters_generator

from jlt.jlt import *
from kmeans_model import *
from pandas_template import inject_into_dataframe
from pca_wrapper import pca_wrapper
from performance_categorizer import *
from cluster_data_gen_settings import *
import pathlib

""" TEST SETTINGS """

num_test_total = 1
num_test_each = 1

std_settings = [100, 500, 1000]
sparsity_settings = [0, 0.33, 0.66, 0.99]
n_settings = [1000, 10000, 30000]
d_settings = [1000, 10000, 30000]
n_d_settings = [(i, j) for i in n_settings for j in d_settings]
a_b_settings = [(-100, 100), (-500, 500), (-1000, 1000)]
num_clusters = [10]
settings = [
    (cluster_data_gen_settings(n=n, d=d, a=a, b=b, std=std, num_clusters=k, num_test_per_cluster=10, sparsity=spa), f"output/kmeans/{std}_{spa}_{(a, b)}")
    for std in std_settings
    for spa in sparsity_settings
    for n,d in n_d_settings
    for a,b in a_b_settings
    for k in num_clusters
]

""" DIM REDUC SETTINGS """
eps = [0.05, 0.1, 0.5, 0.9]
des = [0.05, 0.1]

""" FUNCS """

print([(ep, de) for ep in eps for de in des])

ese_jlt = [jlt_ese("extremely sparse JL transform", ep, de, {"ep": ep, "de": de}) for ep in eps for de in des]
random_jlt = [jlt_r("sparse JL transform", ep, de, {"ep": ep, "de": de}) for ep in eps for de in des]
n_jlt = [jlt("JL transform", ep, de, {"ep": ep, "de": de}) for ep in eps for de in des]
pca = [pca_wrapper("PCA", lambda _: _,  {})]
blank = [dim_reduc_function("Nothing", lambda x: x, {})]
funcs = ese_jlt + random_jlt + n_jlt + pca + blank
num_test_funcs = [num_test_each for i in range(len(ese_jlt) + len(random_jlt) + len(n_jlt))] + [num_test_each] + [num_test_each]

print(ese_jlt)

""" DATA """

for setting in settings:
    print(f"========= Settings = {setting} ==========")
    st, output_path = setting
    n, d, a, b, std, num_clusters, num_test_per_clus, sparsity = st.n, st.d, st.a, st.b, st.std, st.num_clusters, st.num_test_per_cluster, st.sparsity
    for i in range(num_test_total):
        print(f"========= Test No = {i} ==========")
        km = kmeans_model()
        gen = clusters_generator(n, d, a, b, std, num_clusters, num_test_per_clus, sparsity)
        data = gen.generate()
        tester = performance_cat()
        results = tester.performance_test_all(data, km, funcs, num_test_funcs)
        dataframe = inject_into_dataframe(results)
        dataframe.to_csv(output_path, index=False, mode='a', header= not pathlib.Path(output_path).exists())