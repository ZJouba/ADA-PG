from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn.externals import joblib

housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

params = {
    "gamma": reciprocal(0.001, 0.1),
    "C": uniform(0.1, 5)
}

randomS = RandomizedSearchCV(
    SVR(), params, n_iter=50, verbose=1, cv=4, random_state=9)

randomS.fit(X_train_scaled, y_train)

joblib.dump(randomS.best_estimator_, 'randomS_bestE.pkl', compress = 1)