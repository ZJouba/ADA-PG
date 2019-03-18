import os
import pandas as pd
import numpy as np
import pickle
import warnings
import subprocess
import matplotlib.pyplot as plt

from tabulate import tabulate
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials, space_eval
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
from datetime import datetime
from pandas.plotting import scatter_matrix

warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)

currentPath = os.path.dirname(__file__)

trainFile = os.path.join(currentPath, 'train.csv')
testFile = os.path.join(currentPath, 'test.csv')

trainData = pd.read_csv(trainFile, header=0, dtype={'Age': np.float64})
testData = pd.read_csv(testFile, header=0, dtype={'Age': np.float64})
testData.insert(1, column='Survived', value=0)

allData = trainData.append(testData, ignore_index=True)


def dataManipulate(allData):
    import re
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler, Imputer

    def titles(name):
        search = re.search(r' ([A-Za-z]+)\.', name)
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

    allData['FamilySize'] = allData['SibSp'] + allData['Parch'] + 1
    allData['Alone'] = 0
    allData.loc[allData['FamilySize'] == 1, 'Alone'] = 1
    allData['Cabin'] = allData['Cabin'].str[0]
    allData['Cabin'].fillna('U', inplace=True)
    allData['Title'] = allData['Name'].apply(titles)
    allData['Title'] = allData['Title'].map(
        commonTitles).fillna('Rare')
    allData['AgeGroups'] = pd.cut(allData['Age'], 10)

    grouped = allData.groupby(['Sex', 'Pclass', 'Title'])

    allData['Age'] = grouped['Age'].apply(lambda x: x.fillna(x.median()))
    allData.loc[allData['Fare'] == 0.0, 'Fare'] = np.NaN
    allData['Fare'] = grouped['Fare'].apply(
        lambda x: x.fillna(x.median()))

    embarkedMost = allData['Embarked'].value_counts().index[0]

    allData['Embarked'] = allData['Embarked'].fillna(embarkedMost)

    encoder = OneHotEncoder(sparse=False)
    encodedSex = encoder.fit_transform(allData[['Sex']])
    allData['Sex'] = encodedSex
    encoder1 = OneHotEncoder(sparse=False)
    encodedEmbarked = encoder1.fit_transform(allData[['Embarked']])
    allData['Embarked'] = encodedEmbarked
    encoder2 = OneHotEncoder(sparse=False)
    encodedTitle = encoder2.fit_transform(allData[['Title']])
    allData['Title'] = encodedTitle
    encoder3 = OneHotEncoder(sparse=False, categories='auto')
    encodedLevel = encoder3.fit_transform(allData[['Cabin']])
    allData['Cabin'] = encodedLevel

    scaler = StandardScaler()

    allData[['Age']] = scaler.fit_transform(allData[['Age']])
    allData[['Fare']] = scaler.fit_transform(allData[['Fare']])
    allData[['Pclass']] = scaler.fit_transform(allData[['Pclass']])

    pickle.dump(allData, open(os.path.join(
        currentPath, 'preparedData.p'), 'wb'))

    return allData


def tuneHyperparams(finalModel, X_train, y_train):
    global params
    params = {
        # 'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
        # 'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(6, 1), (6, 2)]),
        # 'alpha': hp.uniform('alpha', 0.0001, 0.05),
        'random_state': hp.choice('random_state', np.arange(1, 10, dtype=int))
    }

    def objective(params):
        cv_results = np.sqrt(-cross_val_score(
            finalModel, X_train, y_train, scoring='neg_mean_squared_error', cv=3))

        best_score = max(cv_results)

        loss = 1 - best_score

        return{'loss': loss, 'params': params, 'status': STATUS_OK}

    trials = Trials()
    bestParams = fmin(
        fn=objective,
        space=params,
        algo=tpe.suggest,
        max_evals=500,
        trials=trials
    )

    bestParams = space_eval(params, bestParams)

    pickle.dump(bestParams, open(os.path.join(
        currentPath, 'bestParams.p'), 'wb'))

    return bestParams


def tuneFinal(model, X_train, y_train):
    params = {
        'n_estimators': hp.choice('n_estimators', np.arange(1, 1001, dtype=int)),
        'learning_rate': hp.uniform('learning_rate', 0, 1),
        'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R']),
    }

    def objective(params):
        cv_results = np.sqrt(-cross_val_score(
            model, X_train, y_train, scoring='neg_mean_squared_error', cv=3))

        best_score = max(cv_results)

        loss = 1 - best_score

        return{'loss': loss, 'params': params, 'status': STATUS_OK}

    trials = Trials()
    finalParams = fmin(
        fn=objective,
        space=params,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials
    )

    return finalParams


def trainModel(finalData):
    train, test = train_test_split(finalData, stratify=finalData['Survived'])

    y_train = train.pop('Survived')
    X_train = train

    y_test = test.pop('Survived')
    X_test = test

    A = RandomForestClassifier(
        criterion='gini',
        n_estimators=700,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='auto',
        oob_score=True,
    )
    B = GradientBoostingClassifier(
        criterion='friedman_mse',
        learning_rate=0.075,
        loss='deviance',
        max_depth=3,
        max_features='log2',
        min_samples_leaf=0.1,
        min_samples_split=0.3545454545454546,
        n_estimators=10,
        subsample=1.0
    )
    C = LogisticRegression(
        C=5.1000100000000002,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        max_iter=100,
        multi_class='ovr',
        n_jobs=1,
        penalty='l2',
        solver='liblinear',
        tol=0.0001,
        warm_start=False
    )
    D = MLPClassifier()

    finalModel = VotingClassifier(
        estimators=[('A', A), ('B', B), ('C', C), ('D', D)],
        voting='soft',
    )

    finalModel.fit(X_train, y_train)
    print('\n Cross validation score for untuned: \t',
          max(np.sqrt(-cross_val_score(
              finalModel,
              X_train,
              y_train,
              scoring='neg_mean_squared_error',
              cv=3
          ))))

    y_predict = finalModel.predict(X_test)
    print('Prediction accuracy for untuned: \t',
          accuracy_score(y_predict, y_test))

    print(confusion_matrix(y_predict, y_test))

    """

    finalModel1 = MLPClassifier()

    finalModel = MLPClassifier(
        early_stopping=False,
        solver='lbfgs',
        activation='logistic',
        alpha=1e-6,
        tol=1e-10,
        hidden_layer_sizes=(200, 200),
        learning_rate_init=0.01,
        max_iter=10000,
        random_state=0
    )

    finalModel1.fit(X_train, y_train)
    print('\n Cross validation score for untuned: \t',
          max(np.sqrt(-cross_val_score(
              finalModel,
              X_train,
              y_train,
              scoring='neg_mean_squared_error',
              cv=3
          ))))

    y_predict1 = finalModel.predict(X_test)
    print('Prediction accuracy for untuned: \t',
          accuracy_score(y_predict1, y_test))

    print(confusion_matrix(y_predict1, y_test))


    if input('\n Do you wish to tune the hyperparameters [y/n] \t') == 'y':
        bestParams = tuneHyperparams(finalModel, X_train, y_train)
        # print(bestParams)
    else:
        try:
            bestParams = pickle.load(
                open(os.path.join(currentPath, 'bestParams.p'), 'rb'))
            # print(bestParams)
        except:
            print('\n No tuning done. Tuning now...')
            bestParams = tuneHyperparams(finalModel, X_train, y_train)
            # print(bestParams)

    finalModel = MLPRegressor(
        early_stopping=True,
        # warm_start=True,
        **bestParams,
        # learning_rate='adaptive',
        solver='lbfgs',
        # alpha=0.038631125378018064,
        activation='logistic',
        tol=1e-10,
        hidden_layer_sizes=(200, 200, 200, 200),
        # random_state=9
    )

    finalModel = MLPRegressor(
        hidden_layer_sizes=(6,),
        activation='logistic',
        solver='lbfgs',
    )

    finalModel.fit(X_train, y_train)
    print('\n Cross validation score for tuned: \t', max(np.sqrt(-
                                                                 cross_val_score(finalModel, X_train, y_train, scoring='neg_mean_squared_error', cv=3))))

    y_predict = finalModel.predict(X_test)
    print('Prediction accuracy for tuned: \t \t',
          mean_squared_error(y_predict, y_test))
    """

    pickle.dump(finalModel, open(
        os.path.join(currentPath, 'finalModel.p'), 'wb'))


def chooseModel(data):
    y_train = data.pop('Survived')
    X_train = data

    modellist = (
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        LogisticRegression(),
        MLPClassifier(),
    )

    models = (model.fit(X_train, y_train) for model in modellist)
    scoreList = []

    for model in models:
        name = type(model).__name__
        score = np.sqrt(-cross_val_score(
            model,
            X_train,
            y_train,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=0
        ))
        scoreList = scoreList + \
            ([[str(name), str(score.mean()), str(score.std())]])

    scoreFrame = pd.DataFrame(
        scoreList, columns=['Name', 'Mean', 'Standard Deviation'])
    scoreFrame.sort_values(['Mean', 'Standard Deviation'], ascending=[
                           True, False], inplace=True, )
    print(tabulate(scoreFrame, headers='keys'))


allData = dataManipulate(allData)
allData = pickle.load(open(os.path.join(currentPath, 'preparedData.p'), 'rb'))
allData = allData.drop(['Ticket', 'Name', 'AgeGroups'], axis=1)
trainDataF = allData[:len(trainData)]
testDataF = allData.iloc[len(trainData):]
trainModel(trainDataF)

"""

if os.path.isfile(os.path.join(currentPath, 'preparedData.p')):
    if input('\n Initial data has been prepared previously. Prepare again? [y/n] \t') == 'y':
        allData = dataManipulate(allData)
        allData = allData.drop(['Ticket', 'Name', 'AgeGroups'], axis=1)
        trainDataF = allData[:len(trainData)]
        testDataF = allData.iloc[len(trainData):]
    else:
        allData = pickle.load(open(os.path.join(
            currentPath, 'preparedData.p'), 'rb'))
        allData = allData.drop(['Ticket', 'Name', 'AgeGroups'], axis=1)
        trainDataF = allData[:len(trainData)]
        testDataF = allData.iloc[len(trainData):]
else:
    allData = dataManipulate(allData)
    allData = allData.drop(['Ticket', 'Name', 'AgeGroups'], axis=1)
    trainDataF = allData[:len(trainData)]
    testDataF = allData.iloc[len(trainData):]

if input('\n Do model evaluation? [y/n] \t') == 'y':
    chooseModel(trainDataF)

if os.path.isfile(os.path.join(currentPath, 'finalModel.p')):
    if input('\n Model already exists. Remake? [y/n] \t') == 'y':
        trainModel(trainDataF)
    else:
        lastModel = pickle.load(
            open(os.path.join(currentPath, 'finalModel.p'), 'rb'))
else:
    trainModel(trainDataF)
"""
lastModel = pickle.load(open(os.path.join(currentPath, 'finalModel.p'), 'rb'))

y_test = trainDataF.pop('Survived')
X_test = trainDataF

# print('\n Accuracy for final predictions are: {:.3f}'.format(100 *
#                                                              lastModel.score(X_test, y_test)))

testDataF = testDataF.drop('Survived', axis=1)
survived = lastModel.predict(testDataF)
finalSurvived = pd.DataFrame(testData['PassengerId'])
finalSurvived['Survived'] = survived
finalSurvived.to_csv(os.path.join(
    currentPath, 'submission.csv'), sep=',', index=False)
pathname = os.path.join(
    currentPath, 'submission.csv')

if input('\n Submit? [y/n] \t') == 'y':
    subprocess.Popen(
        ['kaggle competitions submit - c titanic - f ', str(pathname), ' - m', str(datetime)])
