import os
import pandas as pd
import numpy as np
import pickle
import warnings

from tabulate import tabulate
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.exceptions import DataConversionWarning

warnings.simplefilter(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def dataManipulate(allData):
    import re
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer, KBinsDiscretizer, Binarizer
    import seaborn as sea
    import matplotlib.pyplot as plt

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
    allData['FarePerHead'] = allData['Fare']/allData['FamilySize'].astype(int)

    grouped = allData.groupby(['Sex', 'Pclass', 'Title'])
    # allData['Age'] = grouped['Age'].apply(lambda x: x.fillna(x.median()))
    allData.loc[allData['Fare'] == 0.0, 'Fare'] = np.NaN
    allData['Fare'] = grouped['Fare'].apply(
        lambda x: x.fillna(x.median()))

    embarkedMost = allData['Embarked'].value_counts().index[0]
    allData['Embarked'] = allData['Embarked'].fillna(embarkedMost)

    lbinar = LabelBinarizer()
    allData[['Sex']] = lbinar.fit_transform(allData[['Sex']])

    encoder = OneHotEncoder(sparse=False)
    allData['Embarked'] = encoder.fit_transform(allData[['Embarked']])
    allData['Title'] = encoder.fit_transform(allData[['Title']])
    allData['Cabin'] = encoder.fit_transform(allData[['Cabin']])
    allData['Pclass'] = encoder.fit_transform(allData[['Pclass']])

    scaler = StandardScaler()
    allData[['SibSp']] = scaler.fit_transform(allData[['SibSp']])
    allData[['Parch']] = scaler.fit_transform(allData[['Parch']])
    allData[['FamilySize']] = scaler.fit_transform(allData[['FamilySize']])
    allData[['Pclass']] = scaler.fit_transform(allData[['Pclass']])
    allData[['FarePerHead']] = scaler.fit_transform(allData[['FarePerHead']])

    bins = KBinsDiscretizer(encode='onehot-dense', n_bins=3)
    binsFare = bins.fit_transform(allData[['Fare']])
    allData['Fare'] = binsFare

    allData = predictAge(allData)

    binar = Binarizer(threshold=6)
    binarAge = binar.fit_transform(allData[['Age']])
    allData['Age'] = binarAge

    pickle.dump(allData, open(os.path.join(
        currentPath, 'preparedData.p'), 'wb'))

    # sea.pairplot(data=allData[['Fare', 'Survived', 'Age', 'SibSp',
    #                            'Parch', 'Pclass']], hue='Survived')
    # plt.show()

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
        'learning_rate': [0.03, 0.035, 0.04],
        'gamma': [0, 1, 5],
        # 'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        # 'max_depth': [1, 2, 3],
        'n_estimators': [10, 50, 100, 1000, 2000, 5000],
    }

    gridCV = GridSearchCV(finalModel, params, cv=3,
                          scoring='neg_mean_squared_error', verbose=1)

    gridCV.fit(X_train, y_train)

    bestParams = gridCV.best_params_

    print(bestParams)
    pickle.dump(bestParams, open(os.path.join(
        currentPath, 'bestParamsGrid.p'), 'wb'))

    return bestParams


def trainModel(finalData):
    train, test = train_test_split(finalData, stratify=finalData['Survived'])

    y_train = train.pop('Survived')
    X_train = train

    y_test = test.pop('Survived')
    X_test = test

    finalModel = XGBClassifier()

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

    if input('\n Tune? \t') == 'y':
        bestParams = tuneGridCV(finalModel, X_train, y_train)
    else:
        bestParams = pickle.load(
            open(os.path.join(currentPath, 'bestParamsGrid.p'), 'rb'))

    finalModel = XGBClassifier(**bestParams)
    # finalModel = XGBClassifier(colsample_bytree=0.5, gamma=1, learning_rate=0.003, max_depth=3, n_estimators=2000, subsample=1)

    finalModel.fit(X_train, y_train)
    print('\n Cross validation score for tuned: \t',
          max(np.sqrt(-cross_val_score(
              finalModel,
              X_train,
              y_train,
              scoring='neg_mean_squared_error',
              cv=3
          ))))

    y_predict = finalModel.predict(X_test)
    print('Prediction accuracy for tuned: \t',
          accuracy_score(y_predict, y_test))

    print(confusion_matrix(y_predict, y_test))

    pickle.dump(finalModel, open(
        os.path.join(currentPath, 'finalModel.p'), 'wb'))


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

trainData = pd.read_csv(trainFile, header=0, dtype={'Age': np.float64})
testData = pd.read_csv(testFile, header=0, dtype={'Age': np.float64})
testData.insert(1, column='Survived', value=0)

allData = trainData.append(testData, ignore_index=True)

allData = dataManipulate(allData)
allData = allData.drop(
    ['PassengerId', 'Ticket', 'Name', 'AgeGroups', 'Cabin'], axis=1)

trainDataF = allData[:891]
testDataF = allData.iloc[891:]

trainModel(trainDataF)

lastModel = pickle.load(
    open(os.path.join(currentPath, 'finalModel.p'), 'rb'))

y_test = trainDataF.pop('Survived')
X_test = trainDataF

testDataF = testDataF.drop('Survived', axis=1)
survived = lastModel.predict(testDataF)
finalSurvived = pd.DataFrame(testData['PassengerId'])
finalSurvived['Survived'] = survived
finalSurvived.to_csv(os.path.join(
    currentPath, 'submission.csv'), sep=',', index=False)
pathname = os.path.join(
    currentPath, 'submission.csv')

if input('\n Submit? [y/n] \t') == 'y':
    import subprocess
    from datetime import datetime

    now = datetime.now()

    subString = 'kaggle competitions submit - c titanic - f submission.csv - m "' + \
        str(now.strftime("%Y-%m-%d %H:%M") + '"')
    print(subString)
    subprocess.run(subString)
