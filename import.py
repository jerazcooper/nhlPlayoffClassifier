import csv
from sklearn import tree
from IPython.display import Image
import pydotplus
from numpy import mean

with open('Teams.csv', newline='') as f:
    reader = csv.reader(f)
    teams = []
    title = ['Goals', 'Goals Against', 'PIM', 'PP', 'PK']
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

    samples = list(map(lambda r: [r[2], r[3], r[4], r[5], r[6]], teams))
    target = list(map(lambda r: r[7], teams))

    val = []

    for k in range(15):
        test = samples[k * 50:(k * 50) + 50]
        testOut = target[k * 50:(k * 50) + 50]
        samplesK = samples[0:k * 50]
        targetK = target[0:k * 50]
        samplesK.extend(samples[(k * 50) + 50:])
        targetK.extend(target[(k * 50) + 50:])
        clf = tree.DecisionTreeClassifier()
        clf.fit(samplesK, targetK)
        count = 0
        for i, j in zip(clf.predict(test), testOut):
            if i == j:
                count += 1
        val.append((count / len(test)) * 100)

    print(mean(val))
