import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
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

some_digit = X[36000]


# some_digit_image = some_digit.reshape(28, 28)

# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
#            interpolation="nearest")

# plt.axis("off")
# plt.draw()
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(state)
np.random.shuffle(y_train)

KNeigh_class = KNeighborsClassifier()
# KNeigh_class.fit(X_train, y_train)

print(cross_val_score(KNeigh_class, X_train, y_train, cv=3, scoring="accuracy"))

# print("\n Digit predicted as: " + str(KNeigh_class.predict([some_digit])))
# print("\n Digit is: " + str(y[36000]))

param_grid = [
    {'weights': ['uniform', 'distance']},
    {'n_neighbors': [3, 6, 9]},
]

Gridify = GridSearchCV(KNeigh_class, param_grid, cv=5,
                       scoring='neg_mean_squared_error', n_jobs=-1)

Gridify.fit(X_train, y_train)

print(Gridify.best_params_)
print(Gridify.best_estimator_)

print(cross_val_score(Gridify.best_estimator_,
                      X_train, y_train, cv=3, scoring="accuracy"))
