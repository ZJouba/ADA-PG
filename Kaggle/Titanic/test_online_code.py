from tabulate import tabulate
import subprocess
import graphviz
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score, GridSearchCV

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

currentPath = os.path.dirname(__file__)

trainFile = os.path.join(currentPath, 'train.csv')
testFile = os.path.join(currentPath, 'test.csv')

train = pd.read_csv(trainFile).set_index('PassengerId')
test = pd.read_csv(testFile).set_index('PassengerId')
df = pd.concat([train, test], axis=0, sort=False)
df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['IsWomanOrChild'] = ((df.Title == 'Master') | (df.Sex == 'female'))
df['LastName'] = df.Name.str.split(',').str[0]


family = df.groupby(df.LastName).Survived

df['FamilyTotalCount'] = family.transform(lambda s: s[df.IsWomanOrChild].fillna(0).count()) # Count number of women and children in family

df['FamilyTotalCount'] = df.mask(df.IsWomanOrChild, df.FamilyTotalCount - 1, axis=0) # -1 to discount current individual in calculations

df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrChild].fillna(0).sum()) # sum number of women and children survivors in family

df.to_csv(os.path.join(currentPath, 'onlineTest.csv'), sep=',', index=False)

df['FamilySurvivedCount'] = df.mask(df.IsWomanOrChild, df.FamilySurvivedCount - df.Survived.fillna(0), axis=0) # remove survived count from one that survived

df.to_csv(os.path.join(currentPath, 'onlineTest.csv'), sep=',', index=False)

df['FamilySurvivalRate'] = (df.FamilySurvivedCount / df.FamilyTotalCount.replace(0, np.nan)) # calculate family survival rate (only for non-surviving members. surviving members get a 0%)

df.to_csv(os.path.join(currentPath, 'onlineTest.csv'), sep=',', index=False)

df['IsSingleTraveler'] = df.FamilyTotalCount == 0

df.to_csv(os.path.join(currentPath, 'onlineTest.csv'), sep=',', index=False)

x = pd.concat([
    df.FamilySurvivalRate.fillna(0),
    df.IsSingleTraveler,
    df.Sex.replace({'male': 0, 'female': 1}),
], axis=1)

print(x.info())

train_x, test_x = x.loc[train.index], x.loc[test.index]
train_y = df.Survived.loc[train.index]

model = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
model.fit(train_x, train_y)

# tree = graphviz.Source(tree.export_graphviz(model, feature_names=x.columns))

# tree.render(view=True)
