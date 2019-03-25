import os
import pandas as pd
import numpy as np
import pickle
import warnings
import xgboost as xgb

from scipy.stats import boxcox
from tabulate import tabulate
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn import ensemble
from xgboost import XGBClassifier, XGBRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer, KBinsDiscretizer, Binarizer, MinMaxScaler, LabelEncoder
import graphviz
import seaborn as sea
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

warnings.filterwarnings('ignore')


def dataManipulate(allData):
    import re

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
    allData['Title'] = allData['Title'].map(commonTitles).fillna('Rare')
    allData['Surname'] = allData['Name'].apply(surname)
    allData['TicketShort'] = allData['Ticket'].str[:-2]
    allData['FamilySize'] = allData['SibSp'] + allData['Parch'] + 1
    # sea.countplot(allData['FamilySize'], hue=allData['Survived'])
    # plt.show()
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

    scaler = MinMaxScaler()
    allData['Age'] = scaler.fit_transform(allData[['Age']])

    allData['TicketCount'] = allData.groupby('Ticket')['Ticket'].transform(
        lambda x: x.fillna(0).count())
    allData['FeatureX'] = (allData['Fare'] / (allData['TicketCount']))
    allData['FeatureY'] = (allData['FamilySize'] + allData['Age'])

    scaler = StandardScaler()
    allData['FeatureX'] = scaler.fit_transform(allData[['FeatureX']])
    allData['FeatureY'] = scaler.fit_transform(allData[['FeatureY']])

    allData['FeatureX'], lam = boxcox(allData['FeatureX'] + 1)

    allData.to_csv(
        os.path.join(currentPath, 'surnames.csv'), sep=',', index=False
    )

    return allData


def tuneGridCV(finalModel, X_train, y_train, params):
    gridCV = GridSearchCV(finalModel, params, cv=5,
                          scoring='neg_mean_squared_error', verbose=1)

    gridCV.fit(X_train, y_train)

    bestParams = gridCV.best_params_

    print(bestParams)
    print(gridCV.best_estimator_)

    return gridCV.best_estimator_


def trainModel(finalData):
    train, test = train_test_split(
        finalData, test_size=0.25)

    y_train = train.pop('Survived')
    X_train = train

    y_test = test.pop('Survived')
    X_test = test

    # DECISION TREE
    # finalModel = tree.DecisionTreeClassifier(max_depth=1, criterion='entropy')
    # finalModel.fit(X_train, y_train)
    # GRID SEARCH
    # params = {
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [5, 6, 7, 8, 9],
    # }
    # finalModel = tuneGridCV(finalModel, X_train, y_train, params)

    # RANDOM FOREST
    # params = {
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [2, 4, 6, 8, 10],
    #     'n_estimators': [3, 6, 9]
    # }
    # finalModel = ensemble.RandomForestClassifier(oob_score=True)
    # finalModel = tuneGridCV(finalModel, X_train, y_train, params)
    # finalModel = ensemble.RandomForestClassifier(
    #     oob_score=True, max_depth=3, criterion='gini', n_estimators=3)

    # XGBClassifier
    # params = {
    #     'learning_rate': [0.08, 0.09, 0.1],
    # #     'gamma': [0, 1, 5],
    # #     'subsample': [0.8, 0.9, 1],
    # #     'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    #     'max_depth': [1, 2, 3, 4, 5],
    #     'n_estimators': [10, 15, 20],
    # }
    # finalModel = XGBClassifier()
    # # finalModel = XGBClassifier(objective='binary:logistic', eval_metric='error', max_depth=5, eta=0.1, gamma=0.1, colsample_bytree=1, min_child_weight=1)
    # # finalModel = XGBClassifier(colsample_bytree=0.8, gamma=5, learning_rate=0.035, max_depth=3, n_estimators=5000, subsample=0.8)
    # finalModel = tuneGridCV(finalModel, X_train, y_train, params)

    # GRADIENT BOOSTING CLASSIFIER
    finalModel = GradientBoostingClassifier()

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
    print('Model score: \t', finalModel.score(X_test, y_test))

    global thresh
    thresh = 0.9

    # y_predict1 = finalModel.predict_proba(X_test)
    # y_predict = [1. if y > thresh else 0. for y in y_predict1[:, 1]]
    y_predict = finalModel.predict(X_test)
    y_scores = cross_val_predict(finalModel, X_test, y_test, cv=3)
    print('\n Prediction accuracy for untuned: \t',
          accuracy_score(y_predict, y_test))

    print(confusion_matrix(y_test, y_predict))

    fpr, tpr, threshold = roc_curve(y_test, y_scores)

    def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    # plot_roc_curve(fpr, tpr)
    # plt.show()

    # graph = graphviz.Source(tree.export_graphviz(finalModel, feature_names=X_train.columns))
    # graph.render(view=True)

    # featureImportance = pd.DataFrame(
    #     finalModel.feature_importances_, index=X_train.columns, columns=['importance'])
    # featureImportance.sort_values(
    #     by="importance", ascending=False, inplace=True)
    # print('\n', tabulate(featureImportance, headers='keys'), '\n')

    return finalModel


def chooseModel(data):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    y_train = data.pop('Survived')
    X_train = data

    modellist = (
        XGBClassifier(),
        MLPClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
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
cheatFile = os.path.join(currentPath, 'CheatSheet.csv')

cheatFile = pd.read_csv(cheatFile, header=0).set_index('PassengerId').fillna(0)

trainData = pd.read_csv(trainFile, header=0, dtype={
                        'Age': np.float64}).set_index('PassengerId')
testData = pd.read_csv(testFile, header=0, dtype={
                       'Age': np.float64}).set_index('PassengerId')

labels = pd.read_csv(testFile, header=0, dtype={
    'Age': np.float64})

allData = pd.concat([trainData, testData], axis=0, sort=False)

# sea.pairplot(allData, hue='Survived')
# plt.show()

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

allData['Prediction'] = 0
allData['Prediction'][allData['SexGroup'] == 'Woman'] = 1
allData['Prediction'][(allData['SexGroup'] == 'Woman') &
                      (allData['GroupSurvivalRate'] == 0)] = 0
allData['Prediction'][(allData['SexGroup'] == 'Boy') &
                      (allData['GroupSurvivalRate'] == 1)] = 1

allData.loc[893, 'Prediction'] = 1
allData.loc[1251, 'Prediction'] = 0

# trainPlot = allData.iloc[:891]

# trainPlot = trainPlot[trainPlot['GroupID'] != 'Alone']

# plotS = pd.crosstab(trainPlot['Age'], trainPlot['Prediction'])
# plotS.plot.bar(stacked=True, align='center')

# plt.show()

# print('\n Prediction accuracy for manual: \t',
#       accuracy_score(allData.loc[:891, 'Prediction'], allData.loc[:891, 'Survived']))

# TO SUBMIT PREDICTION COLUMN:
manualSub = pd.DataFrame(allData.iloc[891:].pop('Prediction'))
manualSub.rename(columns={'Prediction': 'Survived'}, inplace=True)

# TO SUBMIT MODEL PREDICTIONS:
# modelData = allData[['SexGroup', 'GroupSurvivalRate', 'Prediction']].copy()
modelData = allData[['Survived', 'FeatureX', 'FeatureY', 'GroupID']].copy()

modelData = modelData.loc[modelData['GroupID'] == 'Alone']

# modelData.rename(columns={'Prediction': 'Survived'}, inplace=True)

modelData.to_csv(
    os.path.join(currentPath, 'modelData.csv'), sep=',', index=False
)

modelData = modelData.drop('GroupID', axis=1)

# encoder = LabelEncoder()
# modelData['SexGroup'] = encoder.fit_transform(modelData[['SexGroup']])

# sea.distplot(trainDataF['FeatureX'])
# plt.show()
# sea.distplot(trainDataF['FeatureY'])
# plt.show()

trainDataF = modelData.loc[allData['Survived'].notnull()]

testDataF = modelData[pd.isnull(allData['Survived'])]

lastModel = trainModel(trainDataF)

testDataF = testDataF.drop('Survived', axis=1)
# finalPred = lastModel.predict_proba(testDataF)
# finalPred = [1. if y > thresh else 0. for y in finalPred[:, 1]]
finalPred = lastModel.predict(testDataF)

testDataF['Survived'] = finalPred

modelSub = pd.DataFrame(testDataF.pop('Survived'))

manualSub.to_csv(
    os.path.join(currentPath, 'manual.csv'), sep=',', index=True, header=['Survived']
)

modelSub.to_csv(
    os.path.join(currentPath, 'model.csv'), sep=',', index=True, header=['Survived']
)

manualSub.update(modelSub)


manualSub.to_csv(
    os.path.join(currentPath, 'testing.csv'), sep=',', index=True, header=['Survived']
)

print('Manual accuracy = ', accuracy_score(
    allData.loc[892:, 'Prediction'], cheatFile))
print('Model accuracy = ', accuracy_score(
    manualSub, cheatFile))

# SUBMISSION SECTION
"""
manualSub.to_csv(os.path.join(
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
