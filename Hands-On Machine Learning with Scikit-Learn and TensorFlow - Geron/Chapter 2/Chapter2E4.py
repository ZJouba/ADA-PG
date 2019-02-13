import os
import tarfile
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR
from six.moves import urllib
from scipy import stats

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isfile(os.path.join(housing_path, "housing.csv")):
        if not os.path.isdir(housing_path):
            os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
    else:
        print("\n" + "File already downloaded and extracted" + "\n")

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def idex_of_mostimportant(arr, i):
    return np.sort(np.argpartition(np.array(arr), -i)[-i])

class MostImportantFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, i):
        self.feature_importances = feature_importances
        self.i = i
    def fit(self, X, y=None):
        self.feature_indices_ = index_of_mostimportant(self.feature_importances, self.i)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

housing = load_housing_data()
# print("\n")
# print(housing.head())
# print("\n")

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing_num = housing.drop("ocean_proximity", axis=1)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
    ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)

# print(housing_prepared)
# print(housing_prepared.shape)

supportVectorM = SVR()

param_random = [
    {'kernel': ['linear'], 'C':stats.uniform(5, 100000)},
    {'kernel': ['rbf'], 'C': stats.uniform(5, 100000), 'gamma': stats.uniform(-3, 3)},
]

random_search = RandomizedSearchCV(supportVectorM, param_random, n_iter = 10, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=4, pre_dispatch=2*n_jobs)

random_search.fit(housing_prepared, housing_labels)

n_MSE = random_search.best_score_
RMSE = np.sqrt(-1*n_MSE)
print("+++++++++++++++++++++++++++++++++++++++++++++++\n" + "The root mean squared error, for randomised search, is:")
print(RMSE)
print("\n+++++++++++++++++++++++++++++++++++++++++++++++\n" + "The best parameters determined by the randomised search is:")
print(random_search.best_params_)

feature_importances = random_search.best_estimator_.feature_importances_
i = 5

fullest_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('importantfeaturesonly', MostImportantFeatures(feature_importances, i)),
    ('supportVectorM', SVR(random_search.best_params))
])

housing_prepared_fullest = fullest_pipeline.fit_transform(housing)