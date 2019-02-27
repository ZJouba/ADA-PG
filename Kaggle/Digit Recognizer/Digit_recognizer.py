from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


file = os.path.join(os.path.dirname(__file__), 'train.csv')

data = pd.read_csv(file, nrows=10)

dataset = data.values

y_train = dataset[:, 0]
X_train = dataset[:, 1:]


def showDigit(digit):
    if digit.shape == (784,):
        digit = digit.reshape(28, 28)

    digit_plot = plt.figure(1)
    plt.imshow(digit, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.draw()
    plt.show()


def cnnModel(features, labels):
    # Input layer
    iLayer = tf.reshape(features, [-1, 28, 28, 1])

    # First Convolutional Layer
    cLayer_1 = tf.layers.conv2d(
        inputs=iLayer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    # First Pooling Layer
    pLayer_1 = tf.layers.max_pooling2d(
        inputs=cLayer_1,
        pool_size=[2, 2],
        strides=2
    )


cnnModel(X_train, y_train)

# TODO do grid search on 10, 15 and 20 samples and interpolate training time for HPC
