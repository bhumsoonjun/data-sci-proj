import functools
import os

import numpy as np
import numpy_indexed as npi
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from performance_test_data import performance_test_data


def load_health_news(num_news):
    path = 'data/health+news+in+twitter/Health-Tweets'

    dfs = []
    lens = []
    names = []
    k_clus = num_news
    cnt = 0
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f) and cnt < k_clus:
            dfs.append(pd.read_csv(f, sep="|", on_bad_lines='skip', encoding="latin1").iloc[:, -1])
            lens.append(dfs[-1].size)
            names.append(filename[:-4])
            cnt += 1

    aggregated_df = pd.concat(dfs)
    vectorizer = TfidfVectorizer(stop_words={'english'})
    X = vectorizer.fit_transform(aggregated_df).toarray()
    print(vectorizer.get_feature_names())

    lens = np.array([0] + lens)
    prefix = lens.cumsum()
    labels = np.array(functools.reduce(lambda a, b: a + b, [[names[i]] * lens[i + 1] for i in range(len(names))]))
    means = []
    mean_label_mapping = {}

    for i in range(len(lens) - 1):
        means.append(X[prefix[i]:prefix[i + 1], :].mean(axis=0))
        mean_label_mapping[names[i]] = means[-1]

    train, test, train_labels, test_labels = train_test_split(X, labels, test_size=k_clus*100, shuffle=True)

    group = npi.group_by(test_labels, test)
    naming = np.array(group[0])
    total = np.empty(shape=(1, X.shape[1]))

    for arr in group[1]:
        total = np.concatenate((total, arr), axis=0)

    total = total[1:, :]
    applied = np.apply_along_axis(lambda x: mean_label_mapping[x[0]], axis=1, arr=naming.reshape(-1, 1))
    return performance_test_data(training_data=train, training_label=applied, testing_data=total, testing_label=None)
