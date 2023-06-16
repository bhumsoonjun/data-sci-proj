from health_news_parser import *
from jlt import jlt
from jlt.jlt import *
from kmeans_model import *
from performance_categorizer import *
from sklearn.decomposition import PCA

n = 1000
d = 10000
a = -100
b = 100
cluster_std = 1000
num_cluster = 10
n_test_per_clus = 10
num_test = 1
sparsity = 0.9

""" DIM REDUC SETTINGS """

ep = 0.5
de = 0.5
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
num_test_funcs = [1, 1, 1, 1]

""" MODEL """

kmeans = kmeans_model()

""" DATA """
# cg = clusters_generator(n, d, a, b, cluster_std, num_cluster, n_test_per_clus, sparsity)
# performance_test_data = cg.generate()

performance_test_data = load_health_news(3)

tester = performance_cat()
results = tester.performance_test_all(performance_test_data, kmeans, funcs, num_test_funcs)

for res in results:
    print(res)
