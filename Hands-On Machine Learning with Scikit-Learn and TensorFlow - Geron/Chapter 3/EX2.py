import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

mnist = sio.loadmat(
    r'C:\Users\Zack\scikit_learn_data\mldata\mnist-original.mat')

X, y = mnist["data"], mnist["label"]

X = np.transpose(X)
y = np.transpose(y)
y = np.ravel(y)

some_digit = X[4]
some_digit = some_digit.reshape(28, 28)

X_train, X_test, y_train, y_test = X[0:
                                     10000], X[10000:20000], y[0:10000], y[10000:20000]

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

# plt.imshow(
#     shifted, cmap=matplotlib.cm.Reds, interpolation='nearest', alpha=0.8)
# plt.axis('off')
# plt.draw()
# plt.show()


shifted_images = np.empty([])

for images in X_train:
    for direc in range(1, 4):
        shifted_images = np.append(
            shifted_images, ShiftDigit(X_train[images], direc))

print(X_train.shape)
print(shifted_images.shape)

# KNeigh_class = KNeighborsClassifier(n_neighbors=1, weights='distance')

# print(cross_val_score(KNeigh_class,
#   X_train, y_train, cv = 2, scoring = "accuracy", n_jobs = 7, verbose = 2))
