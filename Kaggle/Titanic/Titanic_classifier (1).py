from __future__ import absolute_import, division, print_function
import warnings

import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import csv

from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, MultiTaskLasso, ElasticNet, MultiTaskElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, LogisticRegression, SGDRegressor
from sklearn.exceptions import DataConversionWarning
from tabulate import tabulate
from hyperopt import hp, tpe, fmin, STATUS_OK

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

train_file = os.path.join(os.path.dirname(__file__), 'train.csv')
test_file = os.path.join(os.path.dirname(__file__), 'test.csv')

train_data = pd.read_csv(train_file)
# test_data = pd.read_csv(test_file)

# train_set = train_data.values
# test_set = test_data.values

# y_train = train_set[:, 0]
# X_train = train_set[:, 1:]

# y_test = test_set[:, 0]
# X_test = test_set[:, 1:]

correlationMatrix = train_data.corr()

Binarizer = LabelBinarizer()
data_sex = train_data['Sex']
data_sex_encoded = Binarizer.fit_transform(data_sex)

# Binarizer1 = LabelBinarizer()
# data_cabin = train_data['Cabin']
# data_cabin = data_cabin.dropna()
# data_cabin = data_cabin.str[0]
# data_cabin[pd.isnull(data_cabin)] = 'NaN'
# data_cabin_encoded = Binarizer1.fit_transform(data_cabin)

# print(Binarizer1.classes_)

train_data_encoded = train_data
train_data_encoded['Sex'] = data_sex_encoded
cabin_fill_data = train_data_encoded.dropna(subset=['Cabin'])
cabin_fill_data['Cabin'] = cabin_fill_data['Cabin'].str[0]
cabin_fill_data[cabin_fill_data['Cabin'] != 'T']
cabin_test_data = train_data_encoded[pd.isnull(train_data['Cabin'])]
test = train_data[pd.isnull(train_data['Cabin'])]
# print(test.shape)
# print(cabin_fill_data.shape)
cabin_test_data = cabin_test_data.drop('Cabin', axis=1)
# print(tabulate(cabin_fill_data, headers='keys'))
cabin_labels = cabin_fill_data['Cabin']
cabin_data = cabin_fill_data.drop(
    ['Cabin', 'Name', 'Ticket', 'Embarked', 'Age'],
    axis=1
)

cabin_test_data = cabin_test_data.drop(
    ['Name', 'Ticket', 'Embarked', 'Age'],
    axis=1
)

cabin_test_data = cabin_test_data.dropna()

Encoder = LabelEncoder()
cabin_labels = Encoder.fit_transform(cabin_labels)
cabin_labels = pd.DataFrame(cabin_labels)

# print(tabulate(cabin_labels.iloc[:2], headers='keys'))
# print(tabulate(cabin_data.iloc[:2], headers='keys'))
# print(tabulate(cabin_test_data.iloc[:2], headers='keys'))


def modelAnalysis():
    global modellist
    modellist = (
        RandomForestRegressor(),
        LinearRegression(),
        Ridge(),
        Lasso(),
        MultiTaskLasso(),
        ElasticNet(),
        MultiTaskElasticNet(),
        Lars(),
        LassoLars(),
        OrthogonalMatchingPursuit(),
        BayesianRidge(),
        ARDRegression(),
        LogisticRegression(),
        SGDRegressor()
    )
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    models = (model.fit(cabin_data, cabin_labels) for model in modellist)
    scoreTable = []
    scoreList = []

    for model in models:
        name = type(model).__name__
        score = np.sqrt(-cross_val_score(model, cabin_data, cabin_labels,
                                         scoring='neg_mean_squared_error', cv=5, n_jobs=4, verbose=0))
        scoreEntry = str(name + ' score is: ' + str(score))
        scoreTable.append(scoreEntry)
        scoreEntry = str(name + ' score mean is: ' + str(score.mean()))
        scoreTable.append(scoreEntry)
        scoreEntry = str(name + ' score std is: ' + str(score.std()))
        scoreTable.append(scoreEntry)
        scoreList = scoreList + ([[str(score.mean()), str(score.std())]])

    scoreFrame = pd.DataFrame(
        scoreList, columns=['Mean', 'Standard Deviation'])
    scoreFrame.sort_values(['Mean', 'Standard Deviation'], ascending=[
                           True, False], inplace=True)
    scoreFrame.to_pickle(os.path.join(
        os.path.dirname(__file__), "./scoreFrame.pkl"))

    with open(os.path.join(os.path.dirname(__file__), 'scores.csv'), 'w') as csvFile:
        writer = csv.writer(csvFile)
        for row in scoreTable:
            writer.writerows([[row]])

    csvFile.close()

    return scoreFrame


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if input('\n Do you wish to evaluate models for shortlisting [y/n] \t') == 'y':
        scoreFrame = modelAnalysis()
    else:
        try:
            scoreFrame = pd.read_pickle(os.path.join(
                os.path.dirname(__file__), "./scoreFrame.pkl"))
        except:
            print(' \n No model analysis done!')
            if input('\n Do you wish to evaluate models for shortlisting [y/n] \t') == 'y':
                scoreFrame = modelAnalysis()


if input('\n Do you wish to tune the hyperparameters [y/n] \t') == 'y':
    ind = scoreFrame.first_valid_index()
    print('\n Top model is ' +
          type(modellist[ind]).__name__)

    print('\n Hyperparameters to tune: ' + str(modellist[ind].get_params()))

    params = {
        'alpha_1': hp.uniform('alpha_1', 1.e-8, 1),
        'alpha_2': hp.uniform('alpha_2', 1.e-8, 1),
        'lambda_1': hp.uniform('lambda_1', 1.e-8, 1),
        'lambda_2': hp.uniform('lambda_2', 1.e-8, 1),
        'threshold_lambda': hp.uniform('threshold_lambda', 1.e+3, 1.e+4),
    }

    def objective(params):
        warnings.filterwarnings(
            action='ignore', category=DataConversionWarning)
        cv_results = np.sqrt(-cross_val_score(modellist[ind], cabin_data, cabin_labels,
                                              scoring='neg_mean_squared_error', cv=10, n_jobs=4, verbose=0))

        best_score = max(cv_results)
        loss = 1 - best_score

        return{'loss': loss, 'params': params, 'status': STATUS_OK}

    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    best = fmin(fn=objective, space=params, algo=tpe.suggest, max_evals=150)

model = ARDRegression(alpha_1=0.8103421714915968, alpha_2=0.12061920957195169,
                      lambda_1=0.15396032701699522, lambda_2=0.4180942054642595, threshold_lambda=7526.009288630043)
# {'alpha_1': 0.8103421714915968, 'alpha_2': 0.12061920957195169, 'lambda_1': 0.15396032701699522, 'lambda_2': 0.4180942054642595, 'threshold_lambda': 7526.009288630043}

model.fit(cabin_data, cabin_labels)
predictions = model.predict(cabin_test_data)

# print(predictions)

rounded = np.round(predictions, decimals=0).astype(int)

filled_cabin = pd.DataFrame(
    Encoder.inverse_transform(rounded))
# print(tabulate(filled_cabin, headers='keys'))
# print(filled_cabin.shape)
cabin_test_data['Cabin'] = filled_cabin
fixed = pd.concat([cabin_fill_data, cabin_test_data], sort=True)
fixed.to_csv(os.path.join(os.path.dirname(__file__), 'fixed.csv'), sep=',')

# parameters = [
#     'Pclass',
#     'Sex',
#     'Age',
#     'Ticket',
#     'Fare',
#     'Embarked',
#     'Cabin',
#     'Survived'
# ]

# # scatter_matrix(train_data_encoded[parameters], figsize=(12, 8))
# # train_data_encoded.plot(kind='scatter', x='Class', y='Fare')
# # train_data_encoded.plot(kind='scatter', x='Class', y='Cabin')
# # train_data_encoded.plot(kind='scatter', x='Fare', y='Cabin')
# plt.figure()
# sea.set(style="darkgrid")
# sea.scatterplot(
#     data=sorted_data,
#     y='Age',
#     x='Cabin'
# )
# plt.figure()
# sea.scatterplot(
#     data=train_data_encoded,
#     y='Pclass',
#     x='Fare'
# )
# plt.figure()
# sea.scatterplot(
#     data=train_data_encoded,
#     y='Pclass',
#     x='Cabin'
# )
# plt.figure()
# sea.scatterplot(
#     data=train_data_encoded,
#     y='Fare',
#     x='Embarked'
# )
# plt.figure()
# sea.scatterplot(
#     data=train_data_encoded,
#     y='Pclass',
#     x='Embarked'
# )
plt.figure()
sea.scatterplot(
    data=fixed,
    y='Cabin',
    x='Survived'
)
# fixed.boxplot('Survived', 'Cabin')
plt.show()

# train_data['age per class'] =


# TODO do grid search on 10, 15 and 20 samples and interpolate training time for HPC
