import csv
from calculate import calculate as c
import printTree
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from numpy import mean
import pandas as pd

YEAR = 1990

with open('Teams.csv', newline='') as f:
    reader = csv.reader(f)
    teams = []
    title = ['Goal Diff', 'PIM', 'PP', 'PK', 'Playoffs']
    for row in reader:
        try:
            row[0] = float(row[0])
            row[16] = float(row[16])
            row[17] = float(row[17])
            row[19] = float(row[19])
            row[21] = float(row[21])
            row[22] = float(row[22])
            row[24] = float(row[24])
            row[25] = float(row[25])
        except ValueError:
            continue

        if (row[0] >= YEAR) & (row[7] == ''):
            x = [row[0], row[2], row[16] - row[17], row[19], row[21] / row[22], 1 - (row[24] / row[25]), False]
            teams.append(x)
        elif (row[0] >= YEAR) & (row[7] != ''):
            x = [row[0], row[2], row[16] - row[17], row[19], row[21] / row[22], 1 - (row[24] / row[25]), True]
            teams.append(x)

    samples = list(map(lambda r: [r[2], r[3], r[4], r[5], r[6]], teams))

    df = pd.DataFrame(samples, columns=title)
    features = df.columns[:4]

    # decision tree
    clf = tree.DecisionTreeClassifier()
    val = []
    for _ in range(0, 30):
        val.append(c(clf=clf, data=df, features=features))
    print('Decision tree', mean(val))
    clf.fit(df[features], df['Playoffs'])
    printTree.display(clf, features)

    # random forest
    clf = RandomForestClassifier(n_jobs=2)
    val = []
    for _ in range(0, 30):
        val.append(c(clf=clf, data=df, features=features))
    print('Random forest', mean(val))

    # Neural net
    clf = clf = MLPClassifier()
    val = []
    for _ in range(0, 30):
        val.append(c(clf=clf, data=df, features=features))
    print('Neural Network', mean(val))
