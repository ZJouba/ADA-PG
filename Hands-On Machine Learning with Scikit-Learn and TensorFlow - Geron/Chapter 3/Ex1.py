import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

mnist = sio.loadmat(
    r'C:\Users\Zack\scikit_learn_data\mldata\mnist-original.mat')

X, y = mnist["data"], mnist["label"]

X = np.transpose(X)
y = np.transpose(y)
y = np.ravel(y)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(state)
np.random.shuffle(y_train)

KNeigh_class = KNeighborsClassifier(n_neighbors=1, weights='distance')

# KNeigh_class.fit(X_train, y_train)
print(cross_val_score(KNeigh_class,
                      X_train, y_train, cv=2, scoring="accuracy", n_jobs=7, verbose=2))

# param_grid = [
#     # {'weights': ['uniform', 'distance']},
#     # {'n_neighbors': [5, 50]},
#     {'n_neighbors': [1, 2, 5]},
# ]

# Gridify = GridSearchCV(KNeigh_class, param_grid, cv=3,
#                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, return_train_score=False)

# Gridify.fit(X_train, y_train)

# print(Gridify.best_params_)
# print(Gridify.best_estimator_)

# print(cross_val_score(Gridify.best_estimator_,
#                       X_train, y_train, cv=3, scoring="accuracy"))
