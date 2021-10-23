import math
import statistics
import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import inv

with open('train.csv', mode='r') as f:
    train_data = []
    for line in f:
        train_matrix = line.strip().split(',')
        train_data.append(train_matrix)

with open('test.csv', mode='r') as f:
    test_data = []
    for line in f:
        test_matrix = line.strip().split(',')
        test_data.append(test_matrix)

sample_Num = len(train_data)  # NO. of samples
X_length = len(train_data[0]) - 1

for row in train_data:
    for i in range(len(train_data[0])):
        row[i] = float(row[i])  # change datatype
for row in test_data:
    for i in range(len(test_data[0])):
        row[i] = float(row[i])


def loss_func(w, dataset):
    loss = 0.5 * sum([(row[-1] - np.inner(w, row[0:7])) ** 2 for row in dataset])
    return loss


def grad(w, dataset):
    grad = []
    for j in range(X_length):
        grad.append(-sum([(row[-1] - np.inner(w, row[0:7])) * row[j] for row in dataset]))
    return grad


def batch_grad(eps, rate, w, dataset):
    loss = []
    while np.linalg.norm(grad(w, dataset)) >= eps:
        loss.append(loss_func(w, dataset))
        w = w - [rate * x for x in grad(w, dataset)]
    return [w, loss]


def sgd_single(eps, rate, w, dataset, pi):
    flag = 0
    loss_vec = []
    for x in pi:
        if np.linalg.norm(sgd_grad(w, pi[x], dataset)) <= eps:
            flag = 1
            return [w, loss_vec, flag]
        loss_vec.append(loss_func(w, dataset))
        w = w - [rate * x for x in sgd_grad(w, pi[x], dataset)]
    return [w, loss_vec, flag]


# sample_idx: evaluate grad at sample index
# grad approximation at sampel_idx
def sgd_grad(w, sample_idx, dataset):
    s_grad = []
    for j in range(X_length):
        s_grad.append(-(dataset[sample_idx][-1] - np.inner(w, dataset[sample_idx][0:7])) * dataset[sample_idx][j])
    return s_grad


def shuffle_sgd(eps, rate, w, dataset, N_epoch):
    loss_all = []
    for i in range(N_epoch):
        pi = np.random.permutation(sample_Num)
        [w, loss_vec, flag] = sgd_single(eps, rate, w, dataset, pi)
        if flag == 1:
            return [w, loss_all]
        loss_all = loss_all + loss_vec
    return [w, loss_all]


passings = 1000
rate1 = 0.01
tolerance1 = 0.001
[ww1, loss1] = batch_grad(tolerance1, rate1, np.zeros(X_length), train_data)
print('Batch Grad')
print('The weight of Batch Grad is: %s.' % ww1)
print('The loss function of train set of Batch Grad is %s.' % loss_func(ww1, train_data))
print('The loss function of test set of Batch Grad is %s.' % loss_func(ww1, test_data))
plot.plot(loss1)
plot.ylabel('loss function value')
plot.xlabel('No. of iterations')
plot.title('Batch Grad \n tolerance= %s, passings = %s ' % (tolerance1, passings))
plot.show()

rate2 = 0.001
tolerance2 = 0.0001
[ww2, loss2] = shuffle_sgd(tolerance2, rate2, np.zeros(X_length), train_data, passings)
print('SGD')
print('The weight of SGD is: %s.' % ww2)
print('The loss function of train set of SGD is %s.' % loss_func(ww2, train_data))
print('The loss function of test set of SGD is %s.' % loss_func(ww2, test_data))
plot.plot(loss2)
plot.ylabel('loss function value')
plot.xlabel('No. of iterations')
plot.title('SGD \n tolerance= %s, passings = %s ' % (tolerance1, passings))
plot.show()

# ---------------------optimal weight----------------
matrix = np.array(train_data)
X = matrix[:, [0, 1, 2, 3, 4, 5, 6]]
Y = matrix[:, -1]
X = X.T
Y = Y.reshape(53, 1)
A = np.dot(X, X.T)
B = np.dot(X, Y)
A_I = np.linalg.inv(A)
w_best = np.dot(A_I, B)
print('The optimal weight is %s.'% w_best)
