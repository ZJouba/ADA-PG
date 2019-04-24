from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


train_file = os.path.join(os.path.dirname(__file__), 'train.csv')
test_file = os.path.join(os.path.dirname(__file__), 'test.csv')
train_data = pd.read_csv(train_file, nrows=100)
test_data = pd.read_csv(test_file)
train_set = train_data.values
test_set = test_data.values
y_train = train_set[:, 0].astype(np.int32)
X_train = train_set[:, 1:] / np.float32(255)
y_test = test_set[:, 0].astype(np.int32)
X_test = test_set[:, 1:] / np.float32(255)


def showDigit(digit):
    if digit.shape == (784,):
        digit = digit.reshape(28, 28)

    digit_plot = plt.figure(1)
    plt.imshow(digit, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.draw()
    plt.show()


def cnnModel(features, labels, mode):
    # Input Layer
    il = tf.reshape(features['x'], [-1, 28, 28, 1])

    # First Level
    # convolutional layer
    cl1 = tf.layers.conv2d(
        inputs=il,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
    )

    # pooling layer
    pl1 = tf.layers.max_pooling2d(
        inputs=cl1,
        pool_size=[2, 2],
        strides=2,
    )

    # Second Level
    # convolutional layer
    cl2 = tf.layers.conv2d(
        inputs=pl1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    # pooling layer
    pl2 = tf.layers.max_pooling2d(
        inputs=cl2,
        pool_size=[2, 2],
        strides=2
    )

    # Third Level
    pl2_flat = tf.reshape(pl2, [-1, 7 * 7 * 64])
    # dense layer
    dl1 = tf.layers.dense(
        inputs=pl2_flat,
        units=1024,
        activation=tf.nn.relu
    )
    drop = tf.layers.dropout(
        inputs=dl1,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # dense layer
    dl2 = tf.layers.dense(
        inputs=drop,
        units=10
    )

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        'classes': tf.argmax(
            input=dl2,
            axis=1
        ),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        'probabilities': tf.nn.softmax(
            dl2,
            name='softmax_tensor'
        )
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    # Loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=dl2
    )

    # Training Optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.001
        )
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )

    # Evaluation metrics
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions['classes']
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


def main():
    #  Estimator
    digitClassifier = tf.estimator.Estimator(
        model_fn=cnnModel,
        model_dir="/tmp/digitCNN"
    )

    # Logging for predictions
    tensorsLog = {'probabilities': 'softmax_tensor'}

    log_hook = tf.train.LoggingTensorHook(
        tensors=tensorsLog,
        every_n_iter=50
    )

    # Training parameters
    train_input = tf.estimator.inputs.numpy_input_fn(
        x={'x': X_train},
        y=y_train,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    # Train
    digitClassifier.train(
        input_fn=train_input,
        steps=10,
        hooks=[log_hook]
    )

    # Evaluation parameters
    eval_input = tf.estimator.inputs.numpy_input_fn(
        x={'x': X_test},
        y=y_test,
        num_epochs=1,
        shuffle=True
    )

    eval_results = digitClassifier.evaluate(
        input_fn=eval_input
    )

    print(eval_results)


main()

# TODO do grid search on 10, 15 and 20 samples and interpolate training time for HPC
