import csv
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

forwards = []
defence = []
with open('NHL07.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            row[6] = float(row[6])    # games played
            row[7] = float(row[7])    # goals
            row[8] = float(row[8])    # assists
            row[10] = float(row[10])  # plus minus
            row[11] = float(row[11])  # penalty minutes
            row[14] = float(row[14])  # time on ice
            row[47] = float(row[47])  # blocks
            row[48] = float(row[48])  # hits
            row[99] = float(row[99])  # nat
        except ValueError:
            continue
        if (row[6] > 20) & ((row[99] == 0) | (row[99] == 2)) & (row[4] != 'D '):
            r = [row[0], row[3], row[4], row[6], row[7], row[8],
                 row[10], row[11], row[14], row[47], row[48]]
            forwards.append(r)
        elif (row[6] > 20) & ((row[99] == 0) | (row[99] == 2)) & (row[4] == 'D '):
            r = [row[0], row[3], row[4], row[6], row[7], row[8],
                 row[10], row[11], row[14], row[47], row[48]]
            defence.append(r)

# Normalize
title = ['Team', 'Player', 'Pos', 'GP', 'Goals', 'Assists', 'PM', 'PIM', 'TOI', 'Blocks', 'Hits']
forwardDf = pd.DataFrame(forwards, columns=title)
for i, f in forwardDf.iterrows():
    forwardDf.set_value(i, 'Goals', 60 * f[4] / f[8])
    forwardDf.set_value(i, 'Assists', 60 * f[5] / f[8])
    forwardDf.set_value(i, 'PM', 60 * f[6] / f[8])
    forwardDf.set_value(i, 'PIM', 60 * f[7] / f[8])
    forwardDf.set_value(i, 'Blocks', 60 * f[9] / f[8])
    forwardDf.set_value(i, 'Hits', 60 * f[10] / f[8])

means = [np.mean(forwardDf['Goals']), np.mean(forwardDf['Assists']),
         np.mean(forwardDf['PM']), np.mean(forwardDf['PIM']),
         np.mean(forwardDf['Blocks']), np.mean(forwardDf['Hits'])]
stdDev = [np.std(forwardDf['Goals']), np.std(forwardDf['Assists']),
          np.std(forwardDf['PM']), np.std(forwardDf['PIM']),
          np.std(forwardDf['Blocks']), np.std(forwardDf['Hits'])]

for i, f in forwardDf.iterrows():
    forwardDf.set_value(i, 'Goals', (f[4] - means[0]) / stdDev[0])
    forwardDf.set_value(i, 'Assists', (f[5] - means[1]) / stdDev[1])
    forwardDf.set_value(i, 'PM', (f[6] - means[2]) / stdDev[2])
    forwardDf.set_value(i, 'PIM', (f[7] - means[3]) / stdDev[3])
    forwardDf.set_value(i, 'Blocks', (f[9] - means[4]) / stdDev[4])
    forwardDf.set_value(i, 'Hits', (f[10] - means[5]) / stdDev[5])

dims = forwardDf[['Goals', 'Assists', 'PM', 'PIM', 'Blocks', 'Hits']]
kmeans = KMeans(n_clusters=4).fit(dims)

centers = kmeans.cluster_centers_
print(kmeans.labels_)

# top-line = 0, second = 1, def = 2, phys = 3
mapping = [0, 0, 0, 0]

maxGoals = 0
goalsIndex = 0
maxBlocks = 0
blocksIndex = 0
maxHits = 0
hitsIndex = 0
for i, center in enumerate(centers):
    if center[0] > maxGoals:
        maxGoals = center[0]
        goalsIndex = i
    if center[4] > maxBlocks:
        maxBlocks = center[4]
        blocksIndex = i
    if center[5] > maxHits:
        maxBlocks = center[5]
        blocksIndex = i

assert goalsIndex != blocksIndex
assert goalsIndex != hitsIndex
assert hitsIndex != blocksIndex

secondIndex = 0
for j in range(4):
    if (j == goalsIndex) | (j == hitsIndex) | (j == blocksIndex):
        continue
    else:
        secondIndex = j
        break


mapping[0] = goalsIndex
mapping[1] = secondIndex
mapping[2] = blocksIndex
mapping[3] = hitsIndex
print(mapping)
