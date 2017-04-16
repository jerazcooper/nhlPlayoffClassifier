import csv
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def normalize(data):
    for i, f in data.iterrows():
        data.set_value(i, 'Goals', 60 * f[4] / f[9])
        data.set_value(i, 'Assists', 60 * f[5] / f[9])
        data.set_value(i, 'Points', 60 * f[6] / f[9])
        data.set_value(i, 'PM', 60 * f[7] / f[9])
        data.set_value(i, 'PIM', 60 * f[8] / f[9])
        data.set_value(i, 'Blocks', 60 * f[10] / f[9])
        data.set_value(i, 'Hits', 60 * f[11] / f[9])

    means = [np.mean(data['Goals']), np.mean(data['Assists']),
             np.mean(data['Points']), np.mean(data['PM']),
             np.mean(data['PIM']), np.mean(data['Blocks']),
             np.mean(data['Hits'])]
    std_dev = [np.std(data['Goals']), np.std(data['Assists']),
               np.std(data['Points']), np.std(data['PM']),
               np.std(data['PIM']), np.std(data['Blocks']),
               np.std(data['Hits'])]

    for i, f in data.iterrows():
        data.set_value(i, 'Goals', (f[4] - means[0]) / std_dev[0])
        data.set_value(i, 'Assists', (f[5] - means[1]) / std_dev[1])
        data.set_value(i, 'Points', (f[6] - means[2]) / std_dev[2])
        data.set_value(i, 'PM', (f[7] - means[3]) / std_dev[3])
        data.set_value(i, 'PIM', (f[8] - means[4]) / std_dev[4])
        data.set_value(i, 'Blocks', (f[10] - means[5]) / std_dev[5])
        data.set_value(i, 'Hits', (f[11] - means[6]) / std_dev[6])

    return data


def import_player_file(file):
    forwards = []
    defence = []
    with open(file, newline='') as f:
        reader = csv.reader(f)
        row1 = next(reader)

        # indices
        team = 0
        name = 0
        pos = 0
        gp = 0
        goals = 0
        assists = 0
        points = 0
        plus = 0
        pim = 0
        toi = 0
        blocks = 0
        hits = 0
        nat = 0

        for index, i in enumerate(row1):
            if (('Tm' in i) | ('Team' in i)) & (team == 0):
                team = index
            elif ((i == 'Player') | ('Name' in i)) & (name == 0):
                name = index
            elif i == 'Pos':
                pos = index
            elif (i == 'GP') & (gp == 0):
                gp = index
            elif (i == 'G') & (goals == 0):
                goals = index
            elif (i == 'A') & (assists == 0):
                assists = index
            elif (i == 'PTS') & (points == 0):
                points = index
            elif i == '+/-':
                plus = index
            elif i == 'PIM':
                pim = index
            elif i == 'TMOI':
                toi = index
            elif i == 'BkS':
                blocks = index
            elif i == 'Hits':
                hits = index
            elif i == 'Nat':
                nat = index

        for row in reader:
            try:
                row[gp] = float(row[gp])  # games played
                row[goals] = float(row[goals])  # goals
                row[assists] = float(row[assists])  # assists
                row[points] = float(row[points])  # points
                row[plus] = float(row[plus])  # plus minus
                row[pim] = float(row[pim])  # penalty minutes
                row[toi] = float(row[toi])  # time on ice
                row[blocks] = float(row[blocks])  # blocks
                row[hits] = float(row[hits])  # hits
                if nat != 0:
                    row[nat] = float(row[nat])  # nat
            except ValueError:
                continue

            r = [row[team], row[name], row[pos], row[gp], row[goals], row[assists], row[points],
                 row[plus], row[pim], row[toi], row[blocks], row[hits]]
            if (row[gp] > 10) & ((nat == 0) | (row[nat] == 0) | (row[nat] == 2)) & ('D' not in row[pos]):
                forwards.append(r)
            elif (row[gp] > 10) & ((nat == 0) | (row[nat] == 0) | (row[nat] == 2)) & ('D' in row[pos]):
                defence.append(r)
    title = ['Team', 'Player', 'Pos', 'GP', 'Goals', 'Assists', 'Points', 'PM', 'PIM', 'TOI', 'Blocks', 'Hits']
    forward_df = normalize(pd.DataFrame(forwards, columns=title))
    dims = forward_df[['Goals', 'Assists', 'PM', 'PIM', 'Blocks', 'Hits']]
    k_means = KMeans(n_clusters=4).fit(dims)
    centers = k_means.cluster_centers_
    # top-line = 0, second = 1, def = 2, phys = 3
    mapping = [0, 0, 0, 0]
    max_goals = 0
    goals_index = 0
    max_blocks = 0
    blocks_index = 0
    max_hits = 0
    hits_index = 0
    max_hits_2 = 0
    hits_index_2 = 0
    for i, center in enumerate(centers):
        if center[0] > max_goals:
            max_goals = center[0]
            goals_index = i
        if center[4] > max_blocks:
            max_blocks = center[4]
            blocks_index = i
        if center[5] > max_hits:
            max_hits_2 = max_hits
            hits_index_2 = hits_index
            max_hits = center[5]
            hits_index = i
        elif center[5] > max_hits_2:
            max_hits_2 = center[5]
            hits_index_2 = i

    if hits_index == blocks_index:
        blocks_index = hits_index_2

    assert goals_index != blocks_index
    assert goals_index != hits_index
    assert hits_index != blocks_index
    second_index = 0
    for j in range(4):
        if (j == goals_index) | (j == hits_index) | (j == blocks_index):
            continue
        else:
            second_index = j
            break
    mapping[goals_index] = 0
    mapping[second_index] = 1
    mapping[blocks_index] = 2
    mapping[hits_index] = 3
    forward_df['Type'] = list(map(lambda x: mapping[x], k_means.labels_))
    defence_df = normalize(pd.DataFrame(defence, columns=title))
    dims = defence_df[['Points', 'PM', 'PIM', 'Blocks', 'Hits']]
    k_means = KMeans(n_clusters=4).fit(dims)
    centers = k_means.cluster_centers_
    # offencive = 0, defencive = 1, average = 2, phys = 3
    mapping = [0, 0, 0, 0]
    max_points = 0
    points_index = 0
    max_blocks = 0
    blocks_index = 0
    max_hits = 0
    hits_index = 0
    max_hits_2 = 0
    hits_index_2 = 0
    for i, center in enumerate(centers):
        if center[0] > max_points:
            max_points = center[0]
            points_index = i
        if center[3] > max_blocks:
            max_blocks = center[3]
            blocks_index = i
        if center[4] > max_hits:
            max_hits_2 = max_hits
            hits_index_2 = hits_index
            max_hits = center[4]
            hits_index = i
        elif center[4] > max_hits_2:
            max_hits_2 = center[4]
            hits_index_2 = i

    if hits_index == blocks_index:
        blocks_index = hits_index_2

    assert points_index != blocks_index
    assert points_index != hits_index
    assert hits_index != blocks_index
    average_index = 0
    for j in range(4):
        if (j == points_index) | (j == hits_index) | (j == blocks_index):
            continue
        else:
            average_index = j
            break
    mapping[points_index] = 0
    mapping[blocks_index] = 1
    mapping[average_index] = 2
    mapping[hits_index] = 3
    defence_df['Type'] = list(map(lambda x: mapping[x], k_means.labels_))
    # print(defence_df)

    return {'forward': forward_df.sort_values(by='GP', ascending=False),
            'defence': defence_df.sort_values(by='GP', ascending=False)}



