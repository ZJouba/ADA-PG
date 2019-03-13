import os
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from tabulate import tabulate
import re
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

trainFile = os.path.join(os.path.dirname(__file__), 'train.csv')
testFile = os.path.join(os.path.dirname(__file__), 'test.csv')

trainData = pd.read_csv(trainFile, header=0, dtype={'Age': np.float64})
testData = pd.read_csv(testFile, header=0, dtype={'Age': np.float64})

allData = [trainData, testData]


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
    dataset['Title'] = dataset['Title'].map(commonTitles).fillna('Rare')

trainData['AgeGroups'] = pd.cut(trainData['Age'], 10)

grouped = trainData.groupby(['Sex', 'Pclass', 'Title'])

trainData['Age'] = grouped['Age'].apply(lambda x: x.fillna(x.median()))
trainData.loc[trainData['Fare'] == 0.0, 'Fare'] = np.NaN
trainData['Fare'] = grouped['Fare'].apply(lambda x: x.fillna(x.median()))

# trainData.loc[trainData['Fare'] == 0.0, 'Fare'] = grouped['Fare'].apply(lambda x: x.median())

embarkedMost = trainData['Embarked'].value_counts().index[0]

trainData['Embarked'].replace(r'^\s+$', embarkedMost, regex=True, inplace=True)

trainData.to_csv(os.path.join(os.path.dirname(__file__), 'full.csv'), sep=',')
