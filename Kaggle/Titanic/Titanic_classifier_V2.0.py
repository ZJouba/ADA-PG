import os
import sys
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import re
import pickle
import csv

from tabulate import tabulate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

currentPath = os.path.dirname(__file__)

trainFile = os.path.join(currentPath, 'train.csv')
testFile = os.path.join(currentPath, 'test.csv')

trainData = pd.read_csv(trainFile, header=0, dtype={'Age': np.float64})
testData = pd.read_csv(testFile, header=0, dtype={'Age': np.float64})
testData.insert(1, column='Survived', value=0)

allData = trainData.append(testData, ignore_index=True)


def dataManipulate(allData):
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
    allData['Level'] = allData['Cabin'].str[0]
    allData['Title'] = allData['Name'].apply(titles)
    allData['Title'] = allData['Title'].map(
        commonTitles).fillna('Rare')
    allData['AgeGroups'] = pd.cut(allData['Age'], 10)

    grouped = allData.groupby(['Sex', 'Pclass', 'Title'])
    allData.to_csv(os.path.join(currentPath, 'allData.csv'),
                   sep=',', index=False)

    allData['Age'] = grouped['Age'].apply(lambda x: x.fillna(x.median()))
    allData.loc[allData['Fare'] == 0.0, 'Fare'] = np.NaN
    allData['Fare'] = grouped['Fare'].apply(
        lambda x: x.fillna(x.median()))

    embarkedMost = allData['Embarked'].value_counts().index[0]

    allData['Embarked'] = allData['Embarked'].fillna(embarkedMost)

    allData.to_csv(os.path.join(
        currentPath, 'allData.csv'), sep=',', index=False)


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
    encoder3 = OneHotEncoder(sparse=False, categories='auto')
    encodedLevel = encoder3.fit_transform(temp[['Level']])
    temp['Level'] = encodedLevel

    return temp


def tuneHyperparams(finalModel, X_train, y_train):
    """params = {
        'n_estimators': hp.choice('n_estimators', np.arange(1, 1001, dtype=int)),
        'min_samples_split': hp.choice('min_samples_split', np.arange(1, 10, dtype=int)),
        'max_features': hp.uniform('max_features', 0.1, 1),
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
        currentPath, 'bestParams.p'), 'wb')) """

    n_estimators = np.arange(450, 550, dtype=int, step=50)
    min_samples_split = np.arange(2, 5, dtype=int)
    min_samples_leaf = np.arange(1, 5, dtype=int)
    
    params = {
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        }

    randomS = GridSearchCV(estimator = finalModel, param_grid = params, cv = 2, n_jobs = -1, verbose=1)
    
    randomS.fit(X_train, y_train)

    bestParams = randomS.best_params_

    return bestParams


def trainModel(finalData):
    finalData = finaldataPreparation(finalData)
    # finalData.pop('PassengerId')

    train, test = train_test_split(finalData, stratify=finalData['Sex'])

    y_train = train.pop('Survived')
    X_train = train

    y_test = test.pop('Survived')
    X_test = test

    finalModel = RandomForestRegressor(
        max_depth=None, max_features='auto', bootstrap=True)

    finalModel.fit(X_train, y_train)
    print('Cross validation score for untuned: \t', max(np.sqrt(-
        cross_val_score(finalModel, X_train, y_train, scoring='neg_mean_squared_error', cv=3))))

    if input('Do you wish to tune the hyperparameters [y/n] \t') == 'y':
        bestParams = tuneHyperparams(finalModel, X_train, y_train)
        print(bestParams)
    else:
        try:
            bestParams = pickle.load(
                open(os.path.join(currentPath, 'bestParams.p'), 'rb'))
            print(bestParams)
        except:
            print('/n No tuning done. Tuning now...')
            bestParams = tuneHyperparams(finalModel, X_train, y_train)
            print(bestParams)

    finalModel = RandomForestClassifier()  # (**bestParams)

    adaFinalModel2 = AdaBoostClassifier(base_estimator=finalModel)

    adaFinalModel2.fit(X_train, y_train)
    print('Cross validation score for tuned: \t', max(np.sqrt(-
        cross_val_score(adaFinalModel2, X_train, y_train, scoring='neg_mean_squared_error', cv=3))))

    y_predict = adaFinalModel2.predict(X_test)
    print('Prediction accuracy for tuned: \t',
          accuracy_score(y_predict, y_test))

    featureImportance = pd.DataFrame(
        adaFinalModel2.feature_importances_, index=X_train.columns, columns=['importance'])
    featureImportance.sort_values(
        by="importance", ascending=False, inplace=True)
    print('\n', tabulate(featureImportance, headers='keys'), '\n')

    """ featuresDrop = featureImportance[featureImportance['importance'] < 0.02]
    featuresDrop = featuresDrop.index.tolist()
    pickle.dump(featuresDrop, open(os.path.join(
        currentPath, 'featuresDrop.p'), 'wb'))

    lastTest = finalData.drop(
        featuresDrop, axis = 1) """

    lastTest = finalData

    lTrain, lTest = train_test_split(
        lastTest, stratify=lastTest['Sex'])

    yTrain = lTrain.pop('Survived')
    Xtrain = lTrain

    yTest = lTest.pop('Survived')
    Xtest = lTest

    adaFinalModel2.fit(Xtrain, yTrain)
    print('Cross validation score for final: \t', max(np.sqrt(-
        cross_val_score(adaFinalModel2, Xtrain, yTrain, scoring='neg_mean_squared_error', cv=3))))

    y_predict = adaFinalModel2.predict(Xtest)
    print('Prediction accuracy for final: \t',
          accuracy_score(y_predict, yTest))

    pickle.dump(adaFinalModel2, open(os.path.join(
        currentPath, 'finalModel.p'), 'wb'))


def chooseModel(data, drop=False):
    data = finaldataPreparation(data)

    if drop:
        featuresDrop = pickle.load(open(os.path.join(
            currentPath, 'featuresDrop.p'), 'rb'))
        data = data.drop(
            featuresDrop, axis=1
        )
        
    train, test = train_test_split(data, stratify=data['Sex'])

    y_train = train.pop('Survived')
    X_train = train

    global modellist
    modellist = (
        RandomForestClassifier(),
        Pipeline((
            ('scaler', StandardScaler()),
            ('SVC', SVC(kernel='poly')),
        )),
        KNeighborsClassifier(),
        GaussianNB(),
        MLPClassifier(solver='adam', max_iter=1000),
    )

    models = (model.fit(X_train, y_train) for model in modellist)
    scoreTable = []
    scoreList = []

    for model in models:
        name = type(model).__name__
        score = np.sqrt(-cross_val_score(model, X_train, y_train,
                                         scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=0))
        scoreEntry = str(name + ' score is: ' + str(score))
        scoreTable.append(scoreEntry)
        scoreEntry = str(name + ' score mean is: ' + str(score.mean()))
        scoreTable.append(scoreEntry)
        scoreEntry = str(name + ' score std is: ' + str(score.std()))
        scoreTable.append(scoreEntry)
        scoreList = scoreList + ([[str(name), str(score.mean()), str(score.std())]])

    scoreFrame = pd.DataFrame(
        scoreList, columns=['Name', 'Mean', 'Standard Deviation'])
    scoreFrame.sort_values(['Mean', 'Standard Deviation'], ascending=[
                           True, False], inplace=True, )
    scoreFrame.to_pickle(os.path.join(
        currentPath, 'scoreFrame.p'))
    print(tabulate(scoreFrame, headers='keys'))


if os.path.isfile(os.path.join(currentPath, 'allData.csv')):
    if input('Data already prepared for level prediction. Redo? [y/n] \t') == 'y':
        dataManipulate(allData)
    else:
        allData = pd.read_csv(os.path.join(
            currentPath, 'allData.csv'), header=0, dtype={'Age': np.float64})
else:
    dataManipulate(allData)


if os.path.isfile(os.path.join(currentPath, 'cabinTrain.p')):
    if input('Train allData already exists. Redo? [y/n] \t') == 'y':

        dataPreparation(allData)
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
    dataPreparation(allData)
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

if os.path.isfile(os.path.join(currentPath, 'allData.p')):
    if input('Model ready data already exists. Redo? [y/n] \t') == 'y':
        encoder = pickle.load(open('encoder.p', 'rb'))

        cabinTrain['Level'] = encoder.inverse_transform(cabinLabel)

        fullData = pd.DataFrame.append(cabinTrain, cabinPredict)
        fullData.sort_values(by="PassengerId", ascending=True, inplace=True)

        trainDataF = fullData[:len(trainData)]
        testDataF = fullData.iloc[len(trainData):]

        pickle.dump(fullData, open(os.path.join(
            currentPath, 'allData.p'), 'wb'))
    else:
        fullData = pickle.load(
            open(os.path.join(currentPath, 'allData.p'), 'rb'))
        trainDataF = fullData[:len(trainData)]
        testDataF = fullData.iloc[len(trainData):]
else:
    encoder = pickle.load(open('encoder.p', 'rb'))

    cabinTrain['Level'] = encoder.inverse_transform(cabinLabel)

    fullData = pd.DataFrame.append(cabinTrain, cabinPredict)
    fullData.sort_values(by="PassengerId", ascending=True, inplace=True)

    trainDataF = fullData[:len(trainData)]
    testDataF = fullData.iloc[len(trainData):]

    pickle.dump(fullData, open(os.path.join(
        currentPath, 'allData.p'), 'wb'))


if os.path.isfile(os.path.join(currentPath, 'finalModel.p')):
    if input('Model already exists. Remake? [y/n] \t') == 'y':
        trainModel(trainDataF)
    else:
        lastModel = pickle.load(
            open(os.path.join(currentPath, 'finalModel.p'), 'rb'))
else:
    trainModel(trainDataF)

lastModel = pickle.load(
    open(os.path.join(currentPath, 'finalModel.p'), 'rb'))

# chooseModel(trainDataF, False)

""" testDataF = finaldataPreparation(testDataF)
featuresDrop = pickle.load(open(os.path.join(
    currentPath, 'featuresDrop.p'), 'rb'))
testDataF = testDataF.drop(
    featuresDrop, axis=1
)
# testDataF = testDataF.drop(
    # ['PassengerId', 'Survived'], axis=1)
testDataF = testDataF.drop('Survived', axis=1)
survived = lastModel.predict(testDataF)

finalSurvived = pd.DataFrame(testData['PassengerId'])
finalSurvived['Survived'] = survived
finalSurvived.to_csv(os.path.join(
    currentPath, 'submission.csv'), sep=',', index=False) """
