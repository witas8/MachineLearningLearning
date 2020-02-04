# unsupervised algorithm = do not feed by labels from data points to train the model
# k = integer that defines number of clusters with centroids
# 1 step: set k centroids in the random k positions and group points to the teams that are the closest to this points
# 2 step: find the centers of the teams and set them as a new centroids (calc an average of these points in each teams)
# 3 step: create new teams that area now the closest to the just calculated centroids
# 4 step: repeat the process until there will be no changes (no moving to another teams)
# times in a loop = number of points * number of centroids * iterations * number of features (or set how many times)
# kmeans = slower but very accurate

import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

#load data
digits = load_digits()
data = scale(digits.data)  # all features, but data is large that is why scale
y = digits.target

# k = len(np.unique(y))  # or just asset for example 10 (classes and their centroids)
k = 10
samples, features = data.shape


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


# we do not have test data, s`o just compare test data LABELS with classifier estimator
# centroids, init position at the begining, n_init = times to generate init centroids, , max_iteration=3000
classifier = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(classifier, "1", data)  # example 1 = name

# read sckit clustering link info
