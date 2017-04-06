import csv
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from IPython.display import Image
import pydotplus
from numpy import mean
import numpy as np
import pandas as pd

with open('Teams.csv', newline='') as f:
    reader = csv.reader(f)
    teams = []
    title = ['Goals', 'Goals Against', 'PIM', 'PP', 'PK', 'Playoffs']
    for row in reader:
        try:
            row[16] = float(row[16])
            row[17] = float(row[17])
            row[19] = float(row[19])
            row[21] = float(row[21])
            row[22] = float(row[22])
            row[24] = float(row[24])
            row[25] = float(row[25])
        except ValueError:
            continue

        if (row[0] >= '1980') & (row[7] == ''):
            x = [row[0], row[2], row[16], row[17], row[19], row[21] / row[22], 1 - (row[24] / row[25]), False]
            teams.append(x)
        elif (row[0] >= '1980') & (row[7] != ''):
            x = [row[0], row[2], row[16], row[17], row[19], row[21] / row[22], 1 - (row[24] / row[25]), True]
            teams.append(x)

    samples = list(map(lambda r: [r[2], r[3], r[4], r[5], r[6], r[7]], teams))
    target = list(map(lambda r: r[7], teams))

    val = []
    for k in range(0, 30):
        df = pd.DataFrame(samples, columns=title)
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= .9
        train, test = df[df['is_train'] == True], df[df['is_train'] == False]
        features = df.columns[:5]
        clf = tree.DecisionTreeClassifier()
        clf.fit(train[features], train['Playoffs'])
        count = 0
        for i, j in zip(clf.predict(test[features]), test['Playoffs']):
            if i == j:
                count += 1
        val.append(count / len(test['Playoffs']))
    print('Decision tree', mean(val))

    val = []
    for k in range(0, 30):
        df = pd.DataFrame(samples, columns=title)
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= .9
        train, test = df[df['is_train'] == True], df[df['is_train'] == False]
        features = df.columns[:5]
        clf = RandomForestClassifier(n_jobs=2)
        clf.fit(train[features], train['Playoffs'])
        count = 0
        for i, j in zip(clf.predict(test[features]), test['Playoffs']):
            if i == j:
                count += 1
        val.append(count / len(test['Playoffs']))
    print('Random forest', mean(val))

    val = []
    for k in range(0, 30):
        df = pd.DataFrame(samples, columns=title)
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= .9
        train, test = df[df['is_train'] == True], df[df['is_train'] == False]
        features = df.columns[:5]
        clf = MLPClassifier()
        clf.fit(train[features], train['Playoffs'])
        count = 0
        for i, j in zip(clf.predict(test[features]), test['Playoffs']):
            if i == j:
                count += 1
        val.append(count / len(test['Playoffs']))
    print('Neural Network', mean(val))
