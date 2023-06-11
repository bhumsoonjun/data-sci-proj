import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import functools

path = 'data/health+news+in+twitter/Health-Tweets'

dfs = []
lens = []
names = []
k_clus = 5
cnt = 0
for filename in os.listdir(path):
    f = os.path.join(path, filename)
    if os.path.isfile(f) and cnt < k_clus:
        dfs.append(pd.read_csv(f, sep="|", on_bad_lines='skip', encoding="latin1").iloc[:, -1])
        lens.append(dfs[-1].size)
        names.append(filename[:-4])
        cnt += 1

lens = np.array([0] + lens)
aggregated_df = pd.concat(dfs)
