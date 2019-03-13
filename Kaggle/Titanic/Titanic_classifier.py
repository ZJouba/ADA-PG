from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib
import matplotlib.pyplot as plt
import os

from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from tabulate import tabulate

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
print(test.shape)
print(cabin_fill_data.shape)
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

rand_reg = RandomForestRegressor(
    n_estimators=5, bootstrap=False)
rand_reg.fit(cabin_data, cabin_labels)

scores = cross_val_score(rand_reg, cabin_data, cabin_labels,
                         scoring='neg_mean_squared_error', cv=3, n_jobs=4)

rmse = np.sqrt(-scores)


def showScores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())


# showScores(rmse)

predictions = rand_reg.predict(cabin_test_data)

# print(predictions)

rounded = np.round(predictions, decimals=0).astype(int)

filled_cabin = pd.DataFrame(Encoder.inverse_transform(rounded))
print(tabulate(filled_cabin,headers='keys'))
# print(filled_cabin.shape)
# print(cabin_test_data.shape)

train_data.loc[train_data['PassengerId']==filled_cabin['PassengerId'],'Cabin'] = filled_cabin['Cabin']
# print(tabulate(test, headers='keys'))

train_data.to_csv(os.path.join(os.path.dirname(__file__), 'fixed.csv'), sep=',')
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
# plt.figure()
# sea.scatterplot(
#     data=train_data_encoded,
#     y='Cabin',
#     x='Embarked'
# )
# # train_data_encoded.boxplot('Survived', 'Cabin')
# plt.show()

# train_data['age per class'] =


# TODO do grid search on 10, 15 and 20 samples and interpolate training time for HPC
