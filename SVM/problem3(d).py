import math
import statistics
import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import minimize
import pandas as pd

df1 = pd.read_csv('train.csv', header=None)
y_1 = df1.iloc[:, -1].values
trainY = 2 * y_1 - 1
trainX = df1.loc[:, 0:df1.shape[1] - 2].values
df2 = pd.read_csv('test.csv', header=None)
testX = df2.loc[:, 0:df2.shape[1] - 2].values
y_2 = df2.iloc[:, -1].values
testY = 2 * y_2 - 1

temp = df1.loc[:, 0:df1.shape[1] - 1].values

for m in range(len(trainX)):
    temp[m][-1] = 1.0
train_data = np.zeros((len(trainX), 6))

for m in range(len(trainX)):
    train_data[m][0:5] = temp[m]
    train_data[m][5] = trainY[m]

temp1 = df2.loc[:, 0:df2.shape[1] - 1].values
for m in range(len(testX)):
    temp1[m][-1] = 1.0
test_data = np.zeros((len(testX), 6))

for m in range(len(testX)):
    test_data[m][0:5] = temp1[m]
    test_data[m][5] = testY[m]


def Gaussian_ker(x1, x2, gamma):
    return np.exp((-np.linalg.norm(x1 - x2) ** 2) / gamma)


def predict_sin(traindata, sample, gamma):
    temp = sum([row[-1] * row[-2] * Gaussian_ker(row[0:5], sample, gamma) for row in traindata])
    return np.sign(temp)


def error_compute(xx, yy):
    cnt = 0
    length = len(xx)
    for i in range(length):
        if xx[i] != yy[i]:
            cnt = cnt + 1
    return cnt / length


def predict(data, traindata, gamma):
    pred_seq = []
    for row in data:
        pred_seq.append(predict_sin(traindata, row[0:5], gamma))
    return pred_seq


def ker_perceptron(traindata, gamma):
    for row in traindata:
        if row[-2] != predict_sin(traindata, row[0:5], gamma):
            row[-1] = row[-1] + 1
    return traindata



Gamma = [ 0.1, 0.5, 1.5, 100]


def main_ker_percp(Gamma):
    for gamma in Gamma:
        print('gamma=', gamma)
        train_up = ker_perceptron(train_data, gamma)
        pred_seq_train = predict(train_data, train_up, gamma)
        pred_seq_test = predict(test_data, train_up, gamma)
        train_label = [row[-2] for row in train_data]
        test_label = [row[-1] for row in test_data]
        err_train = error_compute(pred_seq_train, train_label)
        err_test = error_compute(pred_seq_test, test_label)
        print('train err =', err_train)
        print('test err =', err_test)

main_ker_percp(Gamma)
