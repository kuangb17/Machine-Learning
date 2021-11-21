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


def pred(dual_x, data, gamma):
    label = [row[-1] for row in data]
    pred = []
    for row in data:
        xx = []
        for i in range(len(dual_x)):
            xx.append(dual_x[i] * train_data[i][-1] * Gaussian_ker(train_data[i][0:5], row[0:5], gamma))
        pred_ = np.sign(sum(xx))
        pred.append(pred_)
    return err_(pred, label)


def Gaussian_ker(x1, x2, gamma):
    return np.exp((-np.linalg.norm(x1 - x2) ** 2) / gamma)



def K_(gamma):
    K_hat_t = np.ndarray([len(trainX), len(trainX)])
    for i in range(len(trainX)):
        for j in range(len(trainX)):
            K_hat_t[i, j] = Gaussian_ker(train_data[i][0:5], train_data[j][0:5], gamma)
    return K_hat_t

def svm_obj(x):
    tp = x.dot(K_mat_)
    return 0.5 * tp.dot(x) + (-1 * sum(x))

def constraint(x):
    return np.inner(x, np.asarray(label_))



def svm_dual(C):
    bd = (0, C)
    bds = tuple([bd for i in range(len(trainX))])
    x0 = np.zeros(len(trainX))
    cons = {'type': 'eq', 'fun': constraint}
    sol = minimize(svm_obj, x0, method='SLSQP', bounds=bds, constraints=cons)  # MINIMIZER
    return sol.x


def sup_vec_cnt(dual_x):
    xx = []
    for i in range(len(dual_x)):
        if dual_x[i] != 0.0:
            xx.append(i)
    return [np.count_nonzero(dual_x), set(xx)]


def svm_main(C):
    sol_x = svm_dual(C)
    [cnt, gg] = sup_vec_cnt(sol_x)
    print(cnt)
    print(gg)
    err_1 = pred(sol_x, train_data, gamma)
    err_2 = pred(sol_x, test_data, gamma)
    print('train err=', err_1)
    print('test err=', err_2)
    return [cnt, gg]




# ------------------------- main function ------------------------
# underline at variable name end means global variable
label_ = [row[-1] for row in train_data]
CC_ = [100 / 873, 500 / 873, 700 / 873]
Gamma_ = [ 0.1,0.5,1.5,100]
# =============================================================================
for C in CC_:
    for gamma in Gamma_:
        print('C=',C, 'gamma=', gamma)
        K_mat_ = K_(gamma)
        svm_main(C)
# =============================================================================
# C = 500 / 873
# xx = []
# for gamma in Gamma_:
#     print('C=', C, 'gamma=', gamma)
#     K_mat_ = K_(gamma)
#     [cnt, gg] = svm_main(C)  # gg is the index set of supp. vectors
#     xx.append(gg)
#
# tt = []
# for i in range(len(Gamma_) - 1):
#     tt.append(len(xx[i].intersection(xx[i + 1])))
# print('# overlaps=', tt)

# C =5/873
# gamma = 10
# K_mat_ = K_(gamma)
# svm_main(C)
