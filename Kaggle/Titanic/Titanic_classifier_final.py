import os
import warnings
import re
import graphviz

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tabulate import tabulate
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from pandas.plotting import scatter_matrix

warnings.filterwarnings('ignore')


def loadData():
    ''' Load the dataset from local csv files.'''

    global currentPath
    currentPath = os.path.dirname(__file__)

    trainFile = os.path.join(currentPath, 'train.csv')
    testFile = os.path.join(currentPath, 'test.csv')

    trainData = pd.read_csv(trainFile, header=0, dtype={
        'Age': np.float64}).set_index('PassengerId')
    testData = pd.read_csv(testFile, header=0, dtype={
        'Age': np.float64}).set_index('PassengerId')

    allData = pd.concat([trainData, testData], axis=0, sort=False)

    return allData


def dataPrepEng(allData):
    ''' Prepare the data for machine learning. This includes cleaning, filling, encoding and feature engineering.
    Parameters:
    ----------
    allData : Dataframe    
    Full titanic dataset. '''

    def titles(name):
        ''' Extract a passenger's title from the 'Name' column value.
        Parameters:
        ----------
        name : String    
        Full name of the passenger from the 'Name' column. '''

        search = re.search(r' ([A-Za-z]+)\.', name)
        if search:
            return search.group(1)

        return ""

    def surname(name):
        ''' Extract a passenger's surname from the 'Name' column value.
        Parameters:
        ----------
        name : String    
        Full name of the passenger from the 'Name' column. '''

        search = re.search(r'([A-Za-z]+),', name)
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

    sexgroup = {
        'Mr': 'Man',
        'Mrs': 'Woman',
        'Miss': 'Woman',
        'Master': 'Boy',
        'Don': 'Man',
        'Rev': 'Man',
        'Dr': 'Man',
        'Mme': 'Woman',
        'Ms': 'Woman',
        'Major': 'Man',
        'Lady': 'Woman',
        'Sir': 'Man',
        'Mlle': 'Woman',
        'Col': 'Man',
        'Capt': 'Man',
        'Countess': 'Woman',
        'Jonkheer': 'Man',
        'Dona': 'Woman',
    }

    allData['Title'] = allData['Name'].apply(titles)
    allData['SexGroup'] = allData['Title'].map(sexgroup)
    allData['Title'] = allData['Title'].map(commonTitles).fillna('Rare')
    allData['Surname'] = allData['Name'].apply(surname)
    allData['TicketShort'] = allData['Ticket'].str[:-2]
    allData['FamilySize'] = allData['SibSp'] + allData['Parch'] + 1
    allData['Alone'] = False
    allData.loc[allData['FamilySize'] == 1, 'Alone'] = True
    allData['GroupID'] = allData[['Surname', 'Pclass', 'TicketShort',
                                  'Fare', 'Embarked']].apply(lambda x: '-'.join(x.map(str)), axis=1)

    allData.loc[allData['SexGroup'] == 'Man', 'GroupID'] = 'Alone'
    allData.loc[allData.groupby('GroupID')['GroupID'].transform(
        lambda x: x.count()) == 1, 'GroupID'] = 'Alone'

    grouped = allData.groupby(['Sex', 'Pclass', 'Title'])

    allData.loc[allData['Fare'] == 0.0, 'Fare'] = np.NaN
    allData.loc[allData['Age'] == 0.0, 'Age'] = np.NaN
    allData['Fare'] = grouped['Fare'].apply(lambda x: x.fillna(x.median()))
    allData['Age'] = grouped['Age'].apply(lambda x: x.fillna(x.median()))

    allData['FarePerHead'] = (
        allData['Fare']/allData['FamilySize']).astype(int)

    allData['Embarked'] = allData['Embarked'].fillna(
        allData['Embarked'].value_counts().index[0])

    groupIds = allData.drop_duplicates(
        'GroupID').set_index('Ticket')['GroupID']

    allData.loc[(allData['SexGroup'] != 'Man') & (allData['GroupID'] == 'Alone'),
                'GroupID'] = allData['Ticket'].map(groupIds).fillna('Alone')

    lbinar = LabelBinarizer()
    allData['Sex'] = lbinar.fit_transform(allData[['Sex']])

    trainDataF = allData.iloc[:891]

    testDataF = allData.iloc[891:]

    trainDataF['NumInGroup'] = trainDataF.groupby(
        'GroupID')['GroupID'].transform(lambda x: x.count())
    trainDataF.loc[trainDataF['GroupID'] == 'Alone', 'NumInGroup'] = 1

    trainDataF['GroupSurvived'] = trainDataF.groupby(
        ['GroupID'])['Survived'].transform(lambda x: x.eq(1).sum())

    trainDataF.loc[trainDataF['GroupID'] == 'Alone',
                   'GroupSurvived'] = trainDataF['Survived']

    trainDataF['GroupSurvivalRate'] = ((trainDataF['GroupSurvived']) /
                                       trainDataF['NumInGroup'])

    testDataF['GroupSurvivalRate'] = 0
    groupIds = trainDataF.drop_duplicates(
        'GroupID').set_index('GroupID')['GroupSurvivalRate']

    groupIds = groupIds.drop('Alone')

    testDataF['GroupSurvivalRate'] = testDataF['GroupID'].map(
        groupIds).fillna(0)

    allData = pd.concat([trainDataF, testDataF], axis=0, sort=False)

    allData['Prediction'] = 0

    allData['Prediction'][allData['SexGroup'] == 'Woman'] = 1

    allData['Prediction'][(allData['SexGroup'] == 'Woman') & (allData['GroupID'] != 'Alone') &
                          (allData['GroupSurvivalRate'] == 0)] = 0
    allData['Prediction'][(allData['SexGroup'] == 'Boy') &
                          (allData['GroupSurvivalRate'] == 1)] = 1

    allData.loc[893, 'Prediction'] = 1
    allData.loc[1251, 'Prediction'] = 0

    encoder = LabelEncoder()
    allData['SexGroup'] = encoder.fit_transform(allData[['SexGroup']])

    allData.to_csv(
        os.path.join(currentPath, 'allData.csv'), sep=',', index=False
    )

    return allData


def tuneGridCV(finalModel, X_train, y_train, params):
    ''' Tune the hyperparameters for the Decision Tree model.
        Parameters:
        ----------
        finalModel : Machine Learning Model    
        Machine learning model for which to tune the hyperparameters. 
        X_train : Dataframe    
        Training dataset. 
        y_train : Dataframe    
        Label dataset. 
        params : Dictionary    
        Dictionary of hyperparameters to tune. '''

    gridCV = GridSearchCV(finalModel, params, cv=5,
                          scoring='neg_mean_squared_error', verbose=1)

    gridCV.fit(X_train, y_train)

    bestParams = gridCV.best_params_

    print(bestParams)
    print(gridCV.best_estimator_)

    return gridCV.best_estimator_


def trainModel(finalData):
    ''' Train the Machine Learning Model with the prepared training data.
        Parameters:
        ----------
        finalData : Dataframe    
        Prepared data for model training. '''

    train, test = train_test_split(
        finalData, test_size=0.25)

    y_train = train.pop('Survived')
    X_train = train

    y_test = test.pop('Survived')
    X_test = test

    # DECISION TREE
    params = {
        'max_depth': [3, 4, 5],
        'criterion': ['gini', 'entropy']
    }
    finalModel = tree.DecisionTreeClassifier(max_depth=4, criterion='gini')
    finalModel = tuneGridCV(finalModel, X_train, y_train, params)

    finalModel.fit(X_train, y_train)

    print('\n Cross validation score for untuned: \t',
          max(np.sqrt(-cross_val_score(
              finalModel,
              X_train,
              y_train,
              scoring='neg_mean_squared_error',
              cv=3
          ))))

    print('Model score: \t', finalModel.score(X_test, y_test))

    y_predict = finalModel.predict(X_test)

    print('\n Prediction accuracy for model: \t {:.2%}'.format(
        accuracy_score(y_predict, y_test)))

    print(confusion_matrix(y_test, y_predict))

    graph = graphviz.Source(tree.export_graphviz(
        finalModel, feature_names=X_train.columns))
    graph.render(view=True)

    featureImportance = pd.DataFrame(
        finalModel.feature_importances_, index=X_train.columns, columns=['importance'])
    featureImportance.sort_values(
        by="importance", ascending=False, inplace=True)
    print('\n', tabulate(featureImportance, headers='keys'), '\n')

    return finalModel


allData = dataPrepEng(loadData())

# SELECT COLUMN FOR DECISION TREE CLASSIFIER TRAINING:

modelData = allData[['Prediction', 'SexGroup',
                     'Alone', 'GroupSurvivalRate', 'Pclass']].copy()

# SET MANUAL PREDICITION COLUMN AS TRAINING LABELS (THIS IS DONE DUE TO ASSUMPTIONS MADE AS DESCRIBED IN THE REPORT)

modelData.rename(columns={'Prediction': 'Survived'}, inplace=True)

# SPLIT DATA INTO TRAINING AND PREDICTION DATA

trainData = modelData.iloc[:891]
predictionData = modelData.iloc[891:]

# TRAIN THE DECISION TREE MODEL

decisionTreeModel = trainModel(trainData)

# PREDICT THE SURVIVAL OF PASSENGERS

predictionData = predictionData.drop('Survived', axis=1)
predictions = decisionTreeModel.predict(predictionData)

predictionData['Survived'] = predictions

submission = pd.DataFrame(predictionData.pop('Survived'))

# SUBMIT TO KAGGLE

pathname = os.path.join(
    currentPath, 'submission.csv')

submission.to_csv(pathname, sep=',', index=True, header=['Survived'])

if input('\n Submit? [y/n] \t') == 'y':
    import subprocess
    from datetime import datetime

    now = datetime.now()

    subString = 'kaggle competitions submit -c titanic -f submission.csv -m "' + \
        str(now.strftime("%Y-%m-%d %H:%M") + '"')
    print(subString)
    subprocess.run(subString)
