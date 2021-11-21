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
train_data = np.zeros((len(trainX),6))

for m in range(len(trainX)):
    train_data[m][0:5] = temp[m]
    train_data[m][5] = trainY[m]

temp1 = df2.loc[:, 0:df2.shape[1] - 1].values
for m in range(len(testX)):
    temp1[m][-1] = 1.0
test_data = np.zeros((len(testX),6))

for m in range(len(testX)):
    test_data[m][0:5] = temp1[m]
    test_data[m][5] = testY[m]

def err_(x, y):
    errors = 0

    for ii in range(len(x)):
        if x[ii] != y[ii]:
            errors = errors + 1
    return errors / len(x)


def pred_(wt, data):
    pred = []

    for ii in range(len(data)):
        pred.append(np.sign(np.inner(data[ii][0:len(data[0])-1], wt)))
    label = [row[-1] for row in data]
    return err_(pred, label)


def Gaussian_ker(x1, x2, gamma):
    return np.exp((-np.linalg.norm(x1 - x2) ** 2) / gamma)


# # compute the K_hat matrix
def K_():
    K_hat_1 = np.ndarray([len(trainX), len(trainX)])
    for i in range(len(trainX)):
        for j in range(len(trainX)):
            K_hat_1[i, j] = (trainY[i] * trainY[j] * np.inner(temp[i], temp[j]))
    return K_hat_1


def svm_obj(x):
    tp = x.dot(K_hat)
    return 0.5 * tp.dot(x) + (-1 * sum(x))


def constraint(x):
    return np.inner(x, np.asarray(label_))


def svm_dual(C):
    bd = (0, C)
    bds = tuple([bd for i in range(len(trainX))])
    x0 = np.zeros(len(trainX))
    cons = {'type': 'eq', 'fun': constraint}
    sol = minimize(svm_obj, x0, method='SLSQP', bounds=bds, constraints=cons)
    return sol.x


def w_(s_x):
    ww = []
    for i in range(len(s_x)):
        ww.append(s_x[i] * trainY[i] * np.asarray(train_data[i][0:5]))
    return sum(ww)


def svm_main(C):
    sol_x = svm_dual(C)
    wt = w_(sol_x)
    err_1 = pred_(wt, train_data)
    err_2 = pred_(wt, test_data)
    print('weight=', wt)
    print('train err=', err_1)
    print('test err=', err_2)


# ------------------------- main function ------------------------
# underline at variable name end means global variable

K_hat = K_()
label_ = trainY
CC = [100 / 873, 500 / 873, 700 / 873]
for C in CC:
    svm_main(C)
