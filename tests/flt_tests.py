import numpy as np
from jlt import jlt
from fjlt import fjlt
from utils import utils
from sklearn.decomposition import PCA




data = np.random.randn(1000, 100000)
print("========")
n, d = data.shape
ep = 0.1
k = int(24 / (ep ** 2) * np.log10(n))
proj = fjlt(data.transpose(), k, 0.3)
print(k)
print(proj)
print(proj.shape)
print("=================")
pca = PCA(n_components=10)
pca.fit(data)
p = pca.transform(data)
print(p)