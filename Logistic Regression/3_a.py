import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pandas as pd


def transfer_(X, y):
    data = np.zeros((len(X), 6))
    for i in range(len(X)):
        data[i, 4] = 1
        data[i, 5] = y[i]
        data[i, 0:4] = X[i]
    return data


df1 = pd.read_csv('train.csv', header=None)
y = df1.iloc[:, -1].values
trainY = 2 * y - 1
trainX = df1.loc[:, 0:df1.shape[1] - 2].values
df2 = pd.read_csv('test.csv', header=None)
testX = df2.loc[:, 0:df2.shape[1] - 2].values
y = df2.iloc[:, -1].values
testY = 2 * y - 1

train_data = transfer_(trainX, trainY)
test_data = transfer_(testX, testY)

train_len = len(train_data)
test_len = len(test_data)


def error_(x, y):
    cnt = 0
    length = len(x)
    for i in range(length):
        if x[i] != y[i]:
            cnt = cnt + 1
    return cnt / length


def pred(w, data):
    pred = []
    for i in range(len(data)):
        pred.append(np.sign(np.inner(data[i][0:len(data[0]) - 1], w)))
    label = [row[-1] for row in data]
    return error_(pred, label)


def sigmoid(x):
    if x < -100:
        temp = 0
    else:
        temp = 1 / (1 + np.exp(-1 * x))
    return temp


def loss_fun(w, data, var):
    seq = []
    t1 = 1 / (2 * var) * np.inner(w, w)
    for row in data:
        temp = -row[-1] * np.inner(w, row[0:5])
        if temp > 100:
            t2 = temp
        else:
            t2 = math.log(1 + math.e ** (temp))
        seq.append(t2)
    sum_ = sum(seq)
    return sum_ + t1


# returns an array
def sgd_grad(w, sample, var):
    cc = train_len * sample[-1] * (1 - sigmoid(sample[-1] * np.inner(w, sample[0:5])))
    return np.asarray([w[i] / var - cc * sample[i] for i in range(5)])


def grad(w, var):
    temp = []
    for row in train_data:
        temp.append(row[-1] * (1 - sigmoid(row[-1] * np.inner(w, row[0:5]))) * np.asarray(row[0:5]))
    return w / var - sum(temp)


def GD(w, var, gamma_0, d):
    w = np.asarray(w)
    loss_seq = []
    for i in range(train_len):
        w = w - gamma(i, gamma_0, d) * grad(w, var)
        loss_seq.append(loss_fun(w, train_data, var))
    return [w, loss_seq]


# rate schedule
def gamma(t, gamma_0, d):
    return gamma_0 / (1 + (gamma_0 / d) * t)


def sgd_single(w, perm, var, iter_cnt, gamma_0, d):
    w = np.asarray(w)
    loss_seq = []
    for i in range(train_len):
        w = w - gamma(iter_cnt, gamma_0, d) * sgd_grad(w, train_data[perm[i]], var)
        loss_seq.append(loss_fun(w, train_data, var))
        iter_cnt = iter_cnt + 1
    return [w, loss_seq, iter_cnt]

def sgd_epoch(w, var, T, gamma_0, d):
    iter_cnt = 1
    loss = []
    for i in range(T):
        perm = np.random.permutation(train_len)
        [w, loss_seq, iter_cnt] = sgd_single(w, perm, var, iter_cnt, gamma_0, d)
        loss.extend(loss_seq)
        print('epochs=', i)
    return [w, loss, iter_cnt]


def map_main(VV, TT):
    gamma_0 = 1
    d = 2.0
    train_err = []
    test_err = []
    for var in VV:
        w = np.zeros(5)
        [wt, loss, cnt] = sgd_epoch(w, var, TT, gamma_0, d)
        train_err.append(pred(wt, train_data))
        test_err.append(pred(wt, test_data))
    return [train_err, test_err]


# ---------------------  main function -----------------------------
V = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
T = 100
[train_error, test_error] = map_main(V, T)
print('train_err =', train_error)
print('test_err =', test_error)

w =list(np.zeros(len(train_data[0])-1))
var = 1
[w, loss, iter_cnt] = sgd_epoch(w, var, T, gamma_0, d)
plt.plot(loss[0:100])

