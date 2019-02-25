from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris["data"][:, (2, 3)]
y = iris["target"]


X_with_bias = np.c_[np.ones([len(X), 1]), X]

np.random.seed(42)


def ttSplit(ratio, validation, x_set, y_set):
    ''' Split two datasets x_set and y_set into prescribed ratio. If validation = True, a 20% dataset will also be created.
    Parameters:
    ----------
    ration : float
    Train / Test split ration. 
    validation : bool
    If True a validation set will also be created.
    x_set : ndarray
    Target dataset to be split.
    y_set : ndarray
    Features dataset to be split. '''

    if validation:
        testGroup = int(len(x_set) * ratio)
        validationGroup = int(len(x_set) * 0.2)
        trainGroup = len(x_set) - testGroup - validationGroup
        random_indices = np.random.permutation(len(x_set))

        X_train = x_set[random_indices[:trainGroup]]
        y_train = y_set[random_indices[:trainGroup]]

        X_test = x_set[random_indices[trainGroup:(trainGroup + testGroup)]]
        y_test = y_set[random_indices[trainGroup:(trainGroup + testGroup)]]

        X_valid = x_set[random_indices[(trainGroup + testGroup):]]
        y_valid = y_set[random_indices[(trainGroup + testGroup):]]

        return X_train, X_test, X_valid, y_train, y_test, y_valid

    else:
        testGroup = int(len(x_set) * ratio)
        trainGroup = len(x_set) - testGroup
        random_indices = np.random.permutation(len(x_set))

        X_train = x_set[random_indices[:trainGroup]]
        y_train = y_set[random_indices[:trainGroup]]

        X_test = x_set[random_indices[trainGroup:trainGroup+testGroup]]
        y_test = y_set[random_indices[trainGroup:trainGroup + testGroup]]

        return X_train, X_test, y_train, y_test


def oneHotVector(y):
    '''Converts dataset to One Hot Vector.
    Parameters:
    ----------
    y : ndarray
    Dataset to convert to OHV. '''

    n_classes = y.max() + 1
    m = len(y)
    Y_one_hot = np.zeros((m, n_classes))
    Y_one_hot[np.arange(m), y] = 1
    return Y_one_hot


def softMax(scores):
    '''Calculate the probability that instance belongs to class k using Softmax function.
    Parameters:
    ----------
    scores : ndarray
    Dataset of scores for each class. '''

    top = np.exp(scores)
    bottom = np.sum(top, axis=1, keepdims=True)
    return top/bottom


def logRegress(eta, iters, m):
    '''Trains a Softmax Regression model.
    Parameters:
    ----------
    eta : float
    Learning rate. Default is 0.1. 
    iters : int
    Number of iterations.
    m : int
    Number of instances in dataset.'''

    tiny_value = 1e-10

    stop = np.infty

    param_vec = np.random.randn(n_inputs, n_outputs)

    print('\n Starting training: \n')
    for i in range(iters):
        scores = X_train.dot(param_vec)
        prob = softMax(scores)
        crossEnt = - \
            np.mean(np.sum(y_train_OH * np.log(prob + tiny_value), axis=1))
        error = prob - y_train_OH

        if i % 500 == 0:
            print('Iteration: ' + str(i) +
                  ', Cost function is: ' + str(crossEnt))

        gradients = 1 / m * X_train.T.dot(error)
        param_vec = param_vec - eta * gradients

        predictAccuracy(param_vec, 'valid')

        crossEnt_valid = - \
            np.mean(
                np.sum(y_valid_OH * np.log(predictAccuracy.probs + tiny_value), axis=1))

        if crossEnt_valid < stop:
            stop = crossEnt_valid
        else:
            print('\n Early stopping at iteration: ' + str(i))
            break

    return param_vec


def predictAccuracy(trainedParams, setName):
    '''Calculates the prediction accuracy of model.
    Parameters:
    ----------
    trainedParams : ndarray
    Array of model parameters. 
    setName : str
    Dataset for which to calculate the accuracy.'''

    X_set = globals()['X_' + setName]
    y_set = globals()['y_' + setName]
    scores = X_set.dot(trainedParams)
    predictAccuracy.probs = softMax(scores)
    predictions = np.argmax(predictAccuracy.probs, axis=1)
    predictAccuracy.accuracy = np.mean(predictions == y_set)
    return ('\n Model accuracy is: {:0.2f}'.format(predictAccuracy.accuracy*100) + '%')


X_train, X_test, X_valid, y_train, y_test, y_valid = ttSplit(
    0.2, True, X_with_bias, y)

y_train_OH = oneHotVector(y_train)
y_test_OH = oneHotVector(y_test)
y_valid_OH = oneHotVector(y_valid)

n_inputs = X_train.shape[1]
n_outputs = y_train.max() + 1

model = logRegress(0.1, 5001, len(X_train))
print(predictAccuracy(model, 'valid'))

predictAccuracy(model, 'test')
print('Accuracy on test set: {:0.2f}'.format(
    predictAccuracy.accuracy * 100) + '%')
