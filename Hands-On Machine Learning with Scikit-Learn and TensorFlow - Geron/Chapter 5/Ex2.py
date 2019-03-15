import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_openml
from scipy.stats import uniform, reciprocal
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml('mnist_784', cache=True)

mnist.target = mnist.target.astype(np.int8)

X, y = mnist['data'], mnist['target']

X = StandardScaler().fit_transform(X)

TVratio = 0.05
TTratio = 0.2

train = int(len(X) * (1-TVratio-TTratio))
valid = int(len(X) * TVratio)
test = int(len(X) * TTratio)

X_train = X[:train]
X_valid = X[train: (train + valid)]
X_test = X[(train+valid):]
y_train = y[:train]
y_valid = y[train: (train + valid)]
y_test = y[(train+valid):]


SVC_class = SVC(
    max_iter=5000
)

params = {
    'gamma': reciprocal(0.001, 0.1),
    'C': uniform(1, 10)
}

iters = 100

randomS = RandomizedSearchCV(
    SVC_class,
    params,
    cv=3,
    scoring='neg_mean_squared_error',
    n_iter=iters,
    verbose=2,
    n_jobs=4
)

randomS.fit(X_valid, y_valid)

print(randomS.best_estimator_)
print(randomS.best_score_)

""" SVC(C=4.94664374968174, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001503284486924611,
  kernel='rbf', max_iter=5000, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False) """
