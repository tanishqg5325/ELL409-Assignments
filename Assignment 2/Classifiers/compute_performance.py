import numpy as np


def compute_performance(y_pred, y_test, num_classes):
    performance = {}
    confusion = np.zeros((num_classes, num_classes), dtype=np.int)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion[i, j] = np.sum((y_pred == j) & (y_test == i))
    prec = np.zeros(num_classes)
    rec = np.zeros(num_classes)
    F1 = np.zeros(num_classes)
    for i in range(num_classes):
        if np.sum(confusion[:, i]) == 0:
            prec[i] = 1
        else:
            prec[i] = confusion[i, i] / np.sum(confusion[:, i])
        if np.sum(confusion[i, :]) == 0:
            rec[i] = 1
        else:
            rec[i] = confusion[i, i] / np.sum(confusion[i, :])
        if prec[i] == 0 and rec[i] == 0:
            F1[i] = 0
        else:
            F1[i] = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
    performance['accuracy'] = np.mean(y_pred == y_test)
    performance['precision'] = prec
    performance['recall'] = rec
    performance['F1'] = np.mean(F1)
    return performance

