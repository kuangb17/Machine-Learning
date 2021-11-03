import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Standard_Perceptron(object):

    def __init__(self, learning_rate, T):
        self.learning_rate = learning_rate
        self.T = T
        pass

    def perc_fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1], dtype=float)
        self.errors = []
        for i in range(self.T):
            error = 0
            for xi, target in zip(X, y):
                # net = np.dot(xi, self.w[1:]) + self.w[0]
                # pre_pred = np.where(net >= 0, 1, -1)
                update = self.learning_rate * (target - self.perc_predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                # print(update)
                # print(xi)
                # print(self.w)
                error += int(update != 0.0)
                self.errors.append(error)
                pass
        print('The weight vector is %s' % self.w)

    def net(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def perc_predict(self, X):
        return np.where(self.net(X) >= 0, 1, -1)

    def perc_pred_err(self, X, y):
        count = 0

        for xi, target in zip(X, y):
            if self.perc_predict(xi) != target:
                count += 1
            # net = np.dot(xi, self.w[1:]) + self.w[0]
            # self.pred.append(np.where(net >= 0, 1, -1))
        print('The average prediction error on the test dataset is %s' % (count / len(X)))
        # count = 0
        # for i in range(len(X)):
        #     if pred[i] != X[i][-1]:
        #         count += 1


if __name__ == '__main__':
    df1 = pd.read_csv('train.csv', header=None)
    y = df1.iloc[:, -1].values
    trainY = 2 * y - 1
    trainX = df1.loc[:, 0:df1.shape[1] - 2].values
    df2 = pd.read_csv('test.csv', header=None)
    testX = df2.loc[:, 0:df2.shape[1] - 2].values
    y = df2.iloc[:, -1].values
    testY = 2 * y - 1

    for i in range(1, 11):
        pr = Standard_Perceptron(0.1, i)
        print("T=%s" %i)
        pr.perc_fit(trainX, trainY)
        pr.perc_pred_err(testX, testY)
