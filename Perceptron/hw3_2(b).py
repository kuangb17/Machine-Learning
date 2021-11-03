import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Voted_Perceptron(object):

    def __init__(self, learning_rate, T):
        self.learning_rate = learning_rate
        self.T = T
        self.ww = []
        self.CC = []

        pass


    def perc_fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1], dtype=float)
        k = 0
        C = np.zeros(self.T * len(X)).tolist()
        for i in range(self.T):
            for xi, target in zip(X, y):
                pred_value = np.sign(np.dot(self.w[1:], xi) + self.w[0])
                if pred_value != target:
                    self.w[1:] = self.w[1:] + (target * xi) * self.learning_rate
                    self.w[0] = target * self.learning_rate
                    k += 1
                    row = np.append(self.w[1:], self.w[0])
                    self.ww.append(row)
                    C[k] = 1

                else:
                    C[k] += 1
        for i in range(len(C)):
            if C[i] != 0:
                self.CC.append(C[i])
        print('The weight vectors are %s.' % self.ww)
        print('The number of correctly predicted training examples is %s.' % self.CC)

    def perc_pred_err(self, X, y):
        pred = []
        count = 0
        CC_C = []
        for i in range(len(self.CC)):
            tt = self.ww[i].tolist() + [self.CC[i]]
            CC_C.append(tt)
        # print(CC_C)
        for xi, target in zip(X, y):
            pred.append(np.sign(sum([row[-1] * np.sign(np.dot(row[:4], xi) + row[4]) for row in CC_C])))
        for i in range(len(X)):
            if pred[i] != y[i]:
                count += 1
        print('The average prediction error on the test dataset is %s' %(count / len(X)))

        #         for xi, target in zip(X, y):
        #             np.sign(np.dot(self.ww[:-1], xi) + self.w[-1])
        #             if self.perc_predict(xi) != target:
        #                 count += 1
        #     # net = np.dot(xi, self.w[1:]) + self.w[0]
        #     # self.pred.append(np.where(net >= 0, 1, -1))
        # print('The average prediction error on the test dataset is %s' % (count / len(X)))
        # # count = 0
        # # for i in range(len(X)):
        # #     if pred[i] != X[i][-1]:
        # #         count += 1


if __name__ == '__main__':
    df1 = pd.read_csv('train.csv', header=None)
    y = df1.iloc[:, -1].values
    trainY = 2 * y - 1
    trainX = df1.loc[:, 0:df1.shape[1] - 2].values
    df2 = pd.read_csv('test.csv', header=None)
    testX = df2.loc[:, 0:df2.shape[1] - 2].values
    y = df2.iloc[:, -1].values
    testY = 2 * y - 1
    for i in range(1,11):
        print('T=%s' %i)
        pr = Voted_Perceptron(0.1, i)
        pr.perc_fit(trainX, trainY)
        pr.perc_pred_err(testX, testY)


    # def perc_fit(self, X, y):
    #     k = 0
    #     v = [np.ones_like(X)[0]]
    #     c = [0]
    #     for i in range(self.T):
    #         for j in range(len(X)):
    #             pred_value = 1 if np.dot(v[k], X[i]) > 0 else -1
    #             if pred_value == y[i]:
    #                 c[k] += 1
    #             else:
    #                 v.append(np.add(v[k], np.dot(y[i], X[i])))
    #                 c.append(1)
    #                 k += 1
    #                 print(v)
    #     self.V = v
    #     self.C = c
    #     self.k = k

    # def perc_fit(self, X, y):
    #     c =self.tmp_c
    #     w = self.w =  np.zeros( X.shape[1], dtype=float)
    #     b =self.b
    #     for i in range(self.T):
    #         self.n_err = 0
    #         for xi, target in zip(X, y):
    #             pred_value = np.sign(np.dot(w, xi) + b)
    #             if pred_value != target:
    #                 w = w + target * xi * self.learning_rate
    #                 self.v.append(w)
    #                 self.c.append(c)
    #                 c = 1
    #                 b += target
    #                 self.n_err += 1
    #             else:
    #                 c += 1
    #             print(w)
    #     self.w = w
    #     self.tmp_c = c
    #     self.b = b
    #     print(self.w)



    # def perc_fit(self, X, y):
    #     self.w = np.zeros(1 + X.shape[1], dtype=float)
    #     for i in range(self.T):
    #         for xi, target in zip(X, y):
    #             pred_value = np.sign(np.dot(self.w[1:], xi) + self.w[0])
    #             if pred_value != target:
    #                 self.w[1:] = self.w[1:] + (target * xi) * self.learning_rate
    #                 # self.c.append(c)
    #                 self.w[0] = 1
    #             else:
    #                 self.w[0] += 1
    #     print(self.w)
    #
    #

    # def perc_fit(self, X, y):
    #     self.w = np.zeros(1 + X.shape[1], dtype=float)
    #     c = 1
    #     for i in range(self.T):
    #         for xi, target in zip(X, y):
    #             h = np.dot(xi, self.w[1:]) * target
    #             if h <= 0:
    #                 self.w[1:] = self.w[1:] + target * xi
    #                 self.w[0] = self.w[0] + target
    #                 self.u[1:] = self.u[1:] + target * c * xi
    #                 self.u[0] = self.u[0] + target * c
    #         c = c + 1
    #     self.w[1:] = self.w[1:] - self.u[1:] / c
    #     self.w[0] = np.array([self.w[0] - self.u[0] / c])
    #     print('The weight vector is %s' %self.w)
    #

    # def net(self, X):
    #     return np.dot(X, self.w[1:]) + self.w[0]
    #
    # def perc_predict(self, X):
    #     return np.where(self.net(X) >= 0, 1, -1)
    #
