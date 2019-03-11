import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

dataset = datasets.load_iris()
X = dataset['data'][:, (2, 3)]
y = (dataset['target'] == 2).astype(np.float64)

pipeln = Pipeline((
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss='hinge')),
))

pipeln.fit(X, y)
