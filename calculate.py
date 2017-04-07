import numpy as np


def calculate(clf, data, features):
    data['is_train'] = np.random.uniform(0, 1, len(data)) <= .9
    train, test = data[data['is_train'] == True], data[data['is_train'] == False]
    clf.fit(train[features], train['Playoffs'])
    count = 0
    for i, j in zip(clf.predict(test[features]), test['Playoffs']):
        if i == j:
            count += 1
    return count / len(test['Playoffs'])
