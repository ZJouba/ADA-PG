import os
import sys
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import re
import pickle

from tabulate import tabulate
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

currentPath = os.path.dirname(__file__)

trainFile = os.path.join(currentPath, 'train.csv')
testFile = os.path.join(currentPath, 'test.csv')

trainData = pd.read_csv(trainFile, header=0, dtype={'Age': np.float64})
testData = pd.read_csv(testFile, header=0, dtype={'Age': np.float64})

allData = [trainData, testData]


def dataManipulate(allData):
    def titles(name):
        search = re.search(' ([A-Za-z]+)\.', name)
        if search:
            return search.group(1)
        return ""

    commonTitles = {
        'Mlle': 'Miss',
        'Ms': 'Miss',
        'Miss': 'Miss',
        'Mme': 'Mrs',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Mr': 'Mr'
    }

    for dataset in allData:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['Alone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'Alone'] = 1
        dataset['Level'] = dataset['Cabin'].str[0]
        dataset['Title'] = dataset['Name'].apply(titles)
        dataset['Title'] = dataset['Title'].map(
            commonTitles).fillna('Rare')

    trainData['AgeGroups'] = pd.cut(trainData['Age'], 10)

    grouped = trainData.groupby(['Sex', 'Pclass', 'Title'])

    trainData['Age'] = grouped['Age'].apply(lambda x: x.fillna(x.median()))
    trainData.loc[trainData['Fare'] == 0.0, 'Fare'] = np.NaN
    trainData['Fare'] = grouped['Fare'].apply(
        lambda x: x.fillna(x.median()))

    embarkedMost = trainData['Embarked'].value_counts().index[0]

    trainData['Embarked'] = trainData['Embarked'].fillna(embarkedMost)

    trainData.to_csv(os.path.join(
        currentPath, 'full.csv'), sep=',', index=False)


def dataPreparation(data):
    temp = data

    encoder = OneHotEncoder(sparse=False)
    encodedSex = encoder.fit_transform(temp[['Sex']])
    temp['Sex'] = encodedSex
    encoder1 = OneHotEncoder(sparse=False)
    encodedEmbarked = encoder1.fit_transform(temp[['Embarked']])
    temp['Embarked'] = encodedEmbarked
    encoder2 = OneHotEncoder(sparse=False)
    encodedTitle = encoder2.fit_transform(temp[['Title']])
    temp['Title'] = encodedTitle

    cabinTrain = temp.dropna(subset=['Cabin'], axis=0)

    cabinPredict = temp[pd.isnull(temp['Cabin'])]

    cabinPredict = cabinPredict.drop(
        ['Name', 'Ticket', 'AgeGroups', 'Cabin'], axis=1)

    cabinTrain = cabinTrain.drop(
        ['Name', 'Ticket', 'AgeGroups', 'Cabin'], axis=1)

    cabinLabel = pd.DataFrame(cabinTrain['Level'])

    encoder3 = OneHotEncoder(sparse=False)
    encodedLevel = encoder3.fit_transform(cabinLabel)
    cabinLabel = encodedLevel
    pickle.dump(encoder3, open(os.path.join(
        currentPath, 'encoder.p'), 'wb'))

    cabinTrain = cabinTrain.drop(['Level'], axis=1)

    cabinPredict = cabinPredict.drop(['Level'], axis=1)

    pickle.dump(cabinTrain, open(os.path.join(
        currentPath, 'cabinTrain.p'), 'wb'))

    pickle.dump(cabinLabel, open(os.path.join(
        currentPath, 'cabinLabel.p'), 'wb'))

    pickle.dump(cabinPredict, open(os.path.join(
        currentPath, 'cabinPredict.p'), 'wb'))


def predictCabins(data, labels, predictions):
    model = RandomForestRegressor()

    model.fit(data, labels)

    pickle.dump(model, open(os.path.join(
        currentPath, 'cabinModel.p'), 'wb'))

    predictedLevel = model.predict(predictions)

    predictedLevel = np.round(predictedLevel, decimals=0).astype(int)

    encoder = pickle.load(open('encoder.p', 'rb'))

    predictions['Level'] = encoder.inverse_transform(predictedLevel)

    pickle.dump(predictions, open(os.path.join(
        currentPath, 'cabinPredict.p'), 'wb'))


def finaldataPreparation(data):
    temp = data

    encoder = OneHotEncoder(sparse=False, categories='auto')
    encodedSex = encoder.fit_transform(temp[['Sex']])
    temp['Sex'] = encodedSex
    encoder1 = OneHotEncoder(sparse=False, categories='auto')
    encodedEmbarked = encoder1.fit_transform(temp[['Embarked']])
    temp['Embarked'] = encodedEmbarked
    encoder2 = OneHotEncoder(sparse=False, categories='auto')
    encodedTitle = encoder2.fit_transform(temp[['Title']])
    temp['Title'] = encodedTitle
    encoder3 = OneHotEncoder(sparse=False, categories='auto')
    encodedLevel = encoder3.fit_transform(temp[['Level']])
    temp['Level'] = encodedLevel

    temp = temp.drop(
        ['Name', 'Ticket', 'AgeGroups'], axis=1)

    pickle.dump(temp, open(os.path.join(
        currentPath, 'finalData.p'), 'wb'))

    return temp


if os.path.isfile(os.path.join(currentPath, 'full.csv')):
    if input('Data csv already exists. Redo? [y/n] \t') == 'y':
        dataManipulate(allData)
    else:
        trainData = pd.read_csv(os.path.join(
            currentPath, 'full.csv'), header=0, dtype={'Age': np.float64})
else:
    dataManipulate(allData)


if os.path.isfile(os.path.join(currentPath, 'cabinTrain.p')):
    if input('Train dataset already exists. Redo? [y/n] \t') == 'y':

        dataPreparation(trainData)
        cabinTrain = pickle.load(open(os.path.join(
            currentPath, 'cabinTrain.p'), 'rb'))
        cabinLabel = pickle.load(open(os.path.join(
            currentPath, 'cabinLabel.p'), 'rb'))
        cabinPredict = pickle.load(open(os.path.join(
            currentPath, 'cabinPredict.p'), 'rb'))

    else:

        cabinTrain = pickle.load(open(os.path.join(
            currentPath, 'cabinTrain.p'), 'rb'))
        cabinLabel = pickle.load(open(os.path.join(
            currentPath, 'cabinLabel.p'), 'rb'))
        cabinPredict = pickle.load(open(os.path.join(
            currentPath, 'cabinPredict.p'), 'rb'))
else:
    dataPreparation(trainData)
    cabinTrain = pickle.load(open(os.path.join(
        currentPath, 'cabinTrain.p'), 'rb'))
    cabinLabel = pickle.load(open(os.path.join(
        currentPath, 'cabinLabel.p'), 'rb'))
    cabinPredict = pickle.load(open(os.path.join(
        currentPath, 'cabinPredict.p'), 'rb'))


if os.path.isfile(os.path.join(currentPath, 'cabinPredict.csv')):
    if input('Predictions already made. Remake? [y/n] \t') == 'y':
        predictCabins(cabinTrain, cabinLabel, cabinPredict)
        cabinPredict = pickle.load(
            open(os.path.join(currentPath, 'cabinPredict.p'), 'rb'))
        model = pickle.load(
            open(os.path.join(currentPath, 'cabinModel.p'), 'rb'))
    else:
        cabinPredict = pickle.load(
            open(os.path.join(currentPath, 'cabinPredict.p'), 'rb'))
        model = pickle.load(
            open(os.path.join(currentPath, 'cabinModel.p'), 'rb'))
else:
    predictCabins(cabinTrain, cabinLabel, cabinPredict)
    cabinPredict = pickle.load(
        open(os.path.join(currentPath, 'cabinPredict.p'), 'rb'))
    model = pickle.load(
        open(os.path.join(currentPath, 'cabinModel.p'), 'rb'))

if not os.path.isfile(os.path.join(currentPath, 'readyData.csv')):
    encoder = pickle.load(open('encoder.p', 'rb'))

    cabinTrain['Level'] = encoder.inverse_transform(cabinLabel)

    fullData = pd.DataFrame.append(cabinTrain, cabinPredict)
    fullData.sort_values(by="PassengerId", ascending=True, inplace=True)

    trainData['Level'] = fullData['Level']

    trainData = trainData.drop(['Cabin'], axis=1)

    trainData.to_csv(os.path.join(
        currentPath, 'readyData.csv'), sep=',', index=False)
else:
    trainData = pd.read_csv(os.path.join(
        currentPath, 'readyData.csv'), header=0, dtype={'Age': np.float64})

finalData = finaldataPreparation(trainData)

train, test = train_test_split(finalData)

y_train = train.pop('Survived')
X_train = train

y_test = test.pop('Survived')
X_test = test

finalModel = RandomForestClassifier()

finalModel.fit(X_train, y_train)
print(max(cross_val_score(finalModel, X_train, y_train, scoring='accuracy', cv=3)))

if input('\n Do you wish to tune the hyperparameters [y/n] \t') == 'y':
    params = {
        'n_estimators': hp.choice('n_estimators', np.arange(1, 1001, dtype=int)),
        'min_samples_split': hp.qlognormal('min_samples_split', 2, 1, 1),
        'max_features': hp.uniform('max_features', 0.1, 1),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
    }

    def objective(params):
        cv_results = cross_val_score(
            finalModel, X_train, y_train, scoring='accuracy', cv=3)

        best_score = max(cv_results)

        loss = 1 - best_score

        return{'loss': loss, 'params': params, 'status': STATUS_OK}

    trials = Trials()
    bestParams = fmin(
        fn=objective,
        space=params,
        algo=tpe.suggest,
        max_evals=150,
        trials=trials
    )
    pickle.dump(bestParams, open(os.path.join(
        currentPath, 'bestParams.p'), 'wb'))
    print(bestParams)
else:
    try:
        bestParams = pickle.load(
            open(os.path.join(currentPath, 'bestParams.p'), 'rb'))
        print(bestParams)
    except:
        print('/n No tuning done. Tuning now...')

# model2 = RandomForestClassifier(**bestParams)
model2 = RandomForestClassifier(
    criterion='gini', max_features=0.22931877273925447, min_samples_split=4, n_estimators=878
)

model2.fit(X_train, y_train)
print(max(cross_val_score(model2, X_train, y_train, scoring='accuracy', cv=3)))

y_predict = model2.predict(X_test)
print(accuracy_score(y_predict, y_test))

featureImportance = pd.DataFrame(
    model2.feature_importances_, index=X_train.columns, columns=['importance'])
featureImportance.sort_values(by="importance", ascending=False, inplace=True)
print('\n', tabulate(featureImportance, headers='keys'), '\n')
