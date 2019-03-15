import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

dataset = os.path.join(sys.path[0], "mnist-original.mat")

mnist = sio.loadmat(dataset)

X, y = mnist["data"], mnist["label"]

X = np.transpose(X)
y = np.transpose(y)
y = np.ravel(y)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(state)
np.random.shuffle(y_train)

KNeigh_class = KNeighborsClassifier()

param_grid = [
    {'weights': ['uniform', 'distance']},
    {'n_neighbors': [1, 2, 3, 5, 10]},
]

Gridify = GridSearchCV(KNeigh_class, param_grid, cv=4,
                       scoring='neg_mean_squared_error', n_jobs=16, verbose=2, return_train_score=False)

Gridify.fit(X_train, y_train)

print(Gridify.best_params_)
print(Gridify.best_estimator_)

print(cross_val_score(Gridify.best_estimator_,
                      X_train, y_train, cv=4, scoring="accuracy", n_jobs=16))
