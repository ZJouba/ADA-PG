import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from labellines import labelLines

iris = datasets.load_iris()
X = iris['data'][:, (2, 3)]
y = (iris['target'] == 2).astype(np.float64)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


C = 1.0
iters = 5000
models = (
    LinearSVC(
        C=C,
        loss='hinge',
        max_iter=iters
    ),
    SVC(
        kernel='linear',
        C=C
    ),
    SGDClassifier(
        loss='hinge',
        alpha=(1/(len(X)*C)),
        max_iter=iters,
    )
)

models = (model.fit(X_scaled, y) for model in models)

plt.figure()

meshX, meshY = X_scaled[:, 0], X_scaled[:, 1]
x_min, x_max = (meshX.min() - 1, meshX.max() + 1)
y_min, y_max = (meshY.min() - 1, meshY.max() + 1)

xx = np.linspace(x_min, x_max)

styles = ('k--', 'b:', 'r-')

for model, style in zip(models, styles):
    a = -model.coef_[0, 0] / model.coef_[0, 1]
    b = -model.intercept_[0] / model.coef_[0, 1]
    name = type(model).__name__
    plt.plot(xx, a * xx + b, style, alpha=0.5, label=name)
    plt.scatter(meshX, meshY, c=y, cmap=plt.cm.coolwarm,
                s=20, edgecolors='k')

labelLines(plt.gca().get_lines(), backgroundcolor=(
    1, 1, 1, 0.4), zorder=2)
plt.xlim(xx.min(), xx.max())
plt.ylim(xx.min(), xx.max())
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Model comparison')
plt.show()
