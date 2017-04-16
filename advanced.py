import csv
from calculate import calculate as c
import printTree
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from numpy import mean
import pandas as pd
import classifyPlayers as cp

YEAR = 2007

with open('Teams.csv', newline='') as f:
    reader = csv.reader(f)
    teams = []

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

        if (row[0] >= YEAR) & (row[0] < 2017) & (row[7] == ''):
            x = [row[0], row[3], row[16] - row[17], row[19], row[21] / row[22], 1 - (row[24] / row[25]), False]
            teams.append(x)
        elif (row[0] >= YEAR) & (row[0] < 2017) & (row[7] != ''):
            x = [row[0], row[3], row[16] - row[17], row[19], row[21] / row[22], 1 - (row[24] / row[25]), True]
            teams.append(x)

    players = {'2007': cp.import_player_file('NHL07.csv'),
               '2008': cp.import_player_file('NHL08.csv'),
               '2009': cp.import_player_file('NHL09.csv'),
               '2010': cp.import_player_file('NHL10.csv'),
               '2011': cp.import_player_file('NHL11.csv'),
               '2012': cp.import_player_file('NHL12.csv'),
               '2014': cp.import_player_file('NHL14.csv'),
               '2015': cp.import_player_file('NHL15.csv'),
               '2016': cp.import_player_file('NHL16.csv')}

    advancedTeams = []
    for t in teams:
        p = players[str(int(t[0]))]
        f = p['forward']
        d = p['defence']
        f = f[(f['Team'] == t[1]) | (f['Team'] == t[1].capitalize())]
        d = d[(d['Team'] == t[1]) | (d['Team'] == t[1].capitalize())]
        if (len(f.index) < 12) | (len(d.index) < 6):  # not enough data
            continue
        fTypes = f['Type']
        dTypes = d['Type']
        forwards = [0, 0, 0, 0]
        defence = [0, 0, 0, 0]
        for i, pType in enumerate(fTypes):
            if i < 12:
                forwards[pType] += 1
        for i, pType in enumerate(dTypes):
            if i < 6:
                defence[pType] += 1
        forwards.extend(defence)
        t.extend(forwards)
        advancedTeams.append(t)

    samples = list(map(lambda r: [r[2], r[3], r[4], r[5],
                                  r[7], r[8], r[9], r[10],
                                  r[11], r[12], r[13], r[14], r[6]], advancedTeams))
    title = ['Goal Diff', 'PIM', 'PP', 'PK',
             'Top-line', 'Second-line', 'Def-for', 'Phys-for',
             'Off-def', 'Def-def', 'Av-def', 'Phys-def',
             'Playoffs']
    df = pd.DataFrame(samples, columns=title)

    features = df.columns[:12]

    # decision tree
    clf = tree.DecisionTreeClassifier()
    val = []
    for _ in range(100):
        val.append(c(clf=clf, data=df, features=features))
    print('Decision tree', mean(val))
    clf.fit(df[features], df['Playoffs'])
    printTree.display(clf, features)

    # random forest
    clf = RandomForestClassifier(n_jobs=2)
    val = []
    for _ in range(100):
        val.append(c(clf=clf, data=df, features=features))
    print('Random forest', mean(val))

    # Neural net
    clf = clf = MLPClassifier()
    val = []
    for _ in range(100):
        val.append(c(clf=clf, data=df, features=features))
    print('Neural Network', mean(val))
