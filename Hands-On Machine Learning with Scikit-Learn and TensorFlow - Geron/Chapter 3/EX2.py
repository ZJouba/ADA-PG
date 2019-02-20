import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

dataset = os.path.join(sys.path[0], "mnist-original.mat")

mnist = sio.loadmat(dataset)

X, y = mnist["data"], mnist["label"]

X = np.transpose(X)
y = np.transpose(y)
y = np.ravel(y)

X_train, X_test, y_train, y_test = X[0:
                                     20], X[10000:20000], y[0:20], y[10000:20000]

# some_digit = X[5]
# some_digit = some_digit.reshape(28, 28)

# digit_plot = plt.figure(1)
# plt.imshow(
#     some_digit, cmap=matplotlib.cm.binary, interpolation='nearest')
# plt.axis('off')
# plt.draw()


def ShiftDigit(image, direction):

    temp_image = np.zeros([784, ], dtype=np.int64)
    temp_image = temp_image.reshape(28, 28)

    if direction == '1':
        direction = 1
        horz = 0
    elif direction == '2':
        direction = -1
        horz = 0
    elif direction == '3':
        direction = 0
        horz = 1
    else:
        direction = 0
        horz = -1

    for row in range(28):
        for pixel in range(28):
            try:
                temp_image[row, pixel] = image[(
                    row + horz), (pixel + direction)]
            except:
                temp_image[row, pixel] = 0
    return temp_image


def ShiftedDataset(old_set, old_labels):

    shifted_images = np.empty((0, 784))
    shifted_labels = np.empty((0, 0))

    for images in range(len(old_set)):
        for direc in range(1, 5):
            new_digit = ShiftDigit(old_set[images].reshape(28, 28), direc)
            new_digit = new_digit.reshape(784)
            shifted_images = np.append(shifted_images, [new_digit], axis=0)
            shifted_labels = np.append(shifted_labels, [old_labels[images]])

    new_set = np.concatenate((old_set, shifted_images), axis=0)
    new_labels = np.concatenate((old_labels, shifted_labels), axis=0)

    return new_set, new_labels

# plt.imshow(
#     shifted, cmap=matplotlib.cm.Reds, interpolation='nearest', alpha=0.8)
# plt.axis('off')
# plt.draw()
# plt.show()


if not os.path.isfile(os.path.join(sys.path[0], 'mnist-shifted.npz')):

    X_train_new, y_train_new = ShiftedDataset(X_train, y_train)
    np.savez(os.path.join(
        sys.path[0], 'mnist-shifted.npz'), data=X_train_new, label=y_train_new)

else:

    saved = np.load(
        os.path.join(sys.path[0], 'mnist-shifted.npz'))
    X_train_new, y_train_new = saved['data'], saved['label']
    if not len(X_train_new) == (len(X_train) * 5):
        print("\n Dataset must be updated \n")
        X_train_new, y_train_new = ShiftedDataset(X_train, y_train)
        np.savez(os.path.join(
            sys.path[0], 'mnist-shifted.npz'), data=X_train_new, label=y_train_new)

# some_digit = shifted_images[95]
# some_digit = some_digit.reshape(28, 28)

# digit_plot = plt.figure(1)
# plt.imshow(
#     some_digit, cmap=matplotlib.cm.binary, interpolation='nearest')
# plt.axis('off')
# plt.draw()
# plt.show()

KNeigh_class = KNeighborsClassifier(n_neighbors=1, weights='distance')

print(cross_val_score(KNeigh_class,
                      X_train_new, y_train, cv=2, scoring="accuracy", n_jobs=7, verbose=2))
