import pandas as pd
import numpy as np

from pyod.models.knn import KNN

# the function that gives each data a label of anomaly(1) or not(0)
def knn_by_part(X, r, n, length):
    X = X['value'].values.reshape(-1, 1)
    size = len(X)
    result = [0 for i in range(size)]
    start = 0
    end = start + length
    while start < size and end <= size:
        X_train = X[start: end]
        knn = KNN(contamination=r, n_neighbors = n,n_jobs=-1)
        knn.fit(X_train)
        y_train_pred = knn.labels_
        for i in range(len(y_train_pred)):
            result[start+i] += y_train_pred[i]
        start += 1
        end += 1
    for i in range(size):
        if result[i] >= 1:
            result[i] = 1
        else:
            result[i] = 0
    return result

# output the detected anomalies from dataset
def get_anomalies(data, prediction):
    for i in range(len(prediction)):
        if prediction[i] == 1:
            print(data.iloc[i])
