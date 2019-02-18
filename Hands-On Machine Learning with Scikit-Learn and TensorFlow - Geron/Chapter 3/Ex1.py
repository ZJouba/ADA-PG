import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
import numpy as np

mnist = sio.loadmat(
    r'C:\Users\Zack\scikit_learn_data\mldata\mnist-original.mat')

X, y = mnist["data"], mnist["label"]

X = np.transpose(X)

# some_digit = X[36000]
# some_digit_image = some_digit.reshape(28, 28)

# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
#            interpolation="nearest")

# plt.axis("off")
# plt.draw()
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)

print(shuffle_index)
