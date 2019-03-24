import os
import pandas as pd
import numpy as np
import pickle
import warnings

from scipy.stats import boxcox
from tabulate import tabulate
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn import tree
from sklearn import ensemble
from xgboost import XGBClassifier, XGBRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer, KBinsDiscretizer, Binarizer, MinMaxScaler, LabelEncoder
import graphviz

warnings.simplefilter(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def dataManipulate(allData):
    import re

    import seaborn as sea
    import matplotlib.pyplot as plt

    def titles(name):
        search = re.search(r' ([A-Za-z]+)\.', name)
        if search:
            return search.group(1)
        return ""

    def surname(name):
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
    allData['Surname'] = allData['Name'].apply(surname)
    allData['TicketShort'] = allData['Ticket'].str[:-2]
    allData['FamilySize'] = allData['SibSp'] + allData['Parch'] + 1
    allData['Alone'] = False
    allData.loc[allData['FamilySize'] == 1, 'Alone'] = True
    allData['IsWomanOrChild'] = (
        (allData.Title == 'Master') | (allData.Sex == 'female'))
    allData['SurnameFreq'] = allData.groupby('Surname')['Surname'].transform(
        lambda x: x[allData['IsWomanOrChild']].fillna(0).count())
    allData['GroupID'] = allData[['Surname', 'Pclass', 'TicketShort',
                                  'Fare', 'Embarked']].apply(lambda x: '-'.join(x.map(str)), axis=1)
    allData.loc[allData['SexGroup'] == 'Man', 'GroupID'] = 'Alone'
    allData.loc[allData.groupby('GroupID')['GroupID'].transform(
        lambda x: x.count()) == 1, 'GroupID'] = 'Alone'

    groupIds = allData.drop_duplicates(
        'GroupID').set_index('Ticket')['GroupID']

    allData.loc[(allData['SexGroup'] != 'Man') & (allData['GroupID'] == 'Alone'),
                'GroupID'] = allData['Ticket'].map(groupIds).fillna('Alone')

    lbinar = LabelBinarizer()
    allData['Sex'] = lbinar.fit_transform(allData[['Sex']])

    return allData


def predictAge(allData):
    ageTrainData = allData.loc[allData['Age'].notnull()]
    ageTrainData = ageTrainData.drop(
        ['PassengerId', 'Ticket', 'Name', 'AgeGroups', 'Cabin'], axis=1)
    agePredictLabels = allData[pd.isnull(allData['Age'])]
    agePredictLabels = agePredictLabels.drop(
        ['PassengerId', 'Ticket', 'Name', 'AgeGroups', 'Cabin'], axis=1)

    y_train_age = ageTrainData.pop('Age')
    X_train_age = ageTrainData

    xgClass = XGBClassifier()

    xgClass.fit(X_train_age, y_train_age)

    X_predict_age = agePredictLabels.drop(['Age'], axis=1)

    age_predicted = xgClass.predict(X_predict_age)

    agePredictLabels = allData[pd.isnull(allData['Age'])]
    agePredictLabels['Age'] = age_predicted
    ageTrainData = allData.loc[allData['Age'].notnull()]

    allData = pd.concat([ageTrainData, agePredictLabels])

    allData.sort_values(['PassengerId'], ascending=[True], inplace=True)

    return allData


def tuneGridCV(finalModel, X_train, y_train):
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 3, 4, 5],
    }

    gridCV = GridSearchCV(finalModel, params, cv=5,
                          scoring='neg_mean_squared_error', verbose=1)

    gridCV.fit(X_train, y_train)

    bestParams = gridCV.best_params_

    print(bestParams)
    print(gridCV.best_estimator_)
    # pickle.dump(bestParams, open(os.path.join(
    #     currentPath, 'bestParamsGrid.p'), 'wb'))

    return gridCV.best_estimator_


def trainModel(finalData):
    train, test = train_test_split(
        finalData, test_size=0.15)

    y_train = train.pop('Survived')
    X_train = train

    y_test = test.pop('Survived')
    X_test = test

    # DECISION TREE
    finalModel = tree.DecisionTreeClassifier(max_depth=3, criterion='gini')
    finalModel.fit(X_train, y_train)
        # GRID SEARCH
    # finalModel = tuneGridCV(finalModel, X_train, y_train)

    # RANDOM FOREST
    # finalModel = ensemble.RandomForestClassifier(oob_score=True)
        # finalModel = ensemble.RandomForestClassifier(
    #     oob_score=True, max_depth=3, criterion='entropy', n_estimators=3)

    finalModel.fit(X_train, y_train)
    print('\n Cross validation score for untuned: \t',
          max(np.sqrt(-cross_val_score(
              finalModel,
              X_train,
              y_train,
              scoring='neg_mean_squared_error',
              cv=3
          ))))

    # RANDOM FOREST
    # print('\n OOB score: \t', finalModel.oob_score_)
    # print('Model score: \t', finalModel.score(X_test, y_test))

    y_predict = finalModel.predict(X_test)
    print('\n Prediction accuracy for untuned: \t',
          accuracy_score(y_predict, y_test))

    print(confusion_matrix(y_predict, y_test))

    graph = graphviz.Source(tree.export_graphviz(finalModel, feature_names=X_train.columns))
    graph.render(view=True)

    featureImportance = pd.DataFrame(
        finalModel.feature_importances_, index=X_train.columns, columns=['importance'])
    featureImportance.sort_values(
        by="importance", ascending=False, inplace=True)
    print('\n', tabulate(featureImportance, headers='keys'), '\n')

    return finalModel


def chooseModel(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    y_train = data.pop('Survived')
    X_train = data

    modellist = (
        XGBClassifier(),
        XGBRegressor(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        LogisticRegression(),
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


currentPath = os.path.dirname(__file__)

trainFile = os.path.join(currentPath, 'train.csv')
testFile = os.path.join(currentPath, 'test.csv')

trainData = pd.read_csv(trainFile, header=0, dtype={
                        'Age': np.float64}).set_index('PassengerId')
testData = pd.read_csv(testFile, header=0, dtype={
                       'Age': np.float64}).set_index('PassengerId')

labels = pd.read_csv(testFile, header=0, dtype={
    'Age': np.float64})

allData = pd.concat([trainData, testData], axis=0, sort=False)

allData = dataManipulate(allData)

trainDataF = allData.iloc[:891]
testDataF = allData.iloc[891:]

# trainDataF['SurnameFreq'] = trainDataF.groupby('Surname')['Surname'].transform(lambda x: x[allData['IsWomanOrChild']].fillna(0).count())

trainDataF['NumInGroup'] = trainDataF.groupby(
    'GroupID')['GroupID'].transform(lambda x: x.count())
trainDataF.loc[trainDataF['GroupID'] == 'Alone', 'NumInGroup'] = 1


trainDataF['GroupSurvived'] = trainDataF.groupby(
    ['GroupID'])['Survived'].transform(lambda x: x.eq(1).sum())

trainDataF.loc[trainDataF['GroupID'] == 'Alone',
               'GroupSurvived'] = trainDataF['Survived']


trainDataF['GroupSurvivalRate'] = ((trainDataF['GroupSurvived']) /
                                   trainDataF['NumInGroup'])

trainDataF.to_csv(
    os.path.join(currentPath, 'surnames.csv'), sep=',', index=False
)

testDataF['GroupSurvivalRate'] = 0
groupIds = trainDataF.drop_duplicates(
    'GroupID').set_index('GroupID')['GroupSurvivalRate']

groupIds = groupIds.drop('Alone')

classSurvival = {
    1: 1,
    2: 1,
    3: 0
}


testDataF['GroupSurvivalRate'] = testDataF['GroupID'].map(
    groupIds).fillna(testDataF['Pclass'].map(classSurvival))

allData = pd.concat([trainDataF, testDataF], axis=0, sort=False)

allData.to_csv(
    os.path.join(currentPath, 'surnames.csv'), sep=',', index=False
)


""" allData.loc[(allData['FamilySize'] == 1) & (
    allData['Survived'] == 0), 'FamilySurvivalRate'] = 0


allData['FamilySurvivalRate'][(allData['SexGroup'] == 'Man') & (
    allData['Survived'] == 1)] = 1
allData['FamilySurvivalRate'][(allData['SexGroup'] == 'Man') & (
    allData['Survived'] == 0)] = 0
allData['FamilySurvivalRate'][(allData['SexGroup'] == 'Man') & (
    pd.isna(allData['Survived']))] = 0

allData.loc[allData['FamilySurvivalRate'] < 0, 'FamilySurvivalRate'] = 0 """

allData['Prediction'] = 0
allData['Prediction'][allData['SexGroup'] == 'Woman'] = 1
allData['Prediction'][(allData['SexGroup'] == 'Woman') &
                      (allData['GroupSurvivalRate'] == 0)] = 0
allData['Prediction'][(allData['SexGroup'] == 'Boy') &
                      (allData['GroupSurvivalRate'] == 1)] = 1

print('\n Prediction accuracy for manual: \t',
          accuracy_score(allData.loc[:891,'Prediction'], allData.loc[:891,'Survived']))

# TO SUBMIT PREDICTION COLUMN:
"""submission = allData.iloc[891:].pop('Prediction')"""

# TO SUBMIT MODEL PREDICTIONS:
"""modelData = allData[['SexGroup', 'GroupSurvivalRate', 'Survived']].copy()

modelData.to_csv(
    os.path.join(currentPath, 'modelData.csv'), sep=',', index=False
)

# modelData.rename(columns={'Prediction':'Survived'}, inplace=True)

encoder = LabelEncoder()
modelData['SexGroup'] = encoder.fit_transform(modelData[['SexGroup']])

trainDataF = modelData.iloc[:891]
testDataF = modelData.iloc[891:]

lastModel = trainModel(trainDataF)

testDataF = testDataF.drop('Survived', axis=1)
testDataF['Survived'] = lastModel.predict(testDataF)
submission = testDataF.pop('Survived')"""


# SUBMISSION SECTION
"""
submission.to_csv(os.path.join(
    currentPath, 'submission.csv'), sep=',', index=True, header=['Survived'])
pathname = os.path.join(
    currentPath, 'submission.csv')

if input('\n Submit? [y/n] \t') == 'y':
    import subprocess
    from datetime import datetime

    now = datetime.now()

    subString = 'kaggle competitions submit -c titanic -f submission.csv -m "' + \
        str(now.strftime("%Y-%m-%d %H:%M") + '"')
    print(subString)
    subprocess.run(subString)
"""
