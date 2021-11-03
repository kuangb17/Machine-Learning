import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Average_Perceptron(object):

    def __init__(self, learning_rate, T):
        self.learning_rate = learning_rate
        self.T = T
        pass

    def perc_fit(self, X, y):
        self.w= self.u = np.zeros(1 + X.shape[1], dtype=float)
        c = 1
        for i in range(self.T):
            for xi, target in zip(X, y):
                h = np.dot(xi, self.w[1:]) * target
                if h <= 0:
                    self.w[1:] = self.w[1:] + target * xi * self.learning_rate
                    self.w[0] = self.w[0] + target * self.learning_rate
                    self.u[1:] = self.u[1:] + target * c * xi * self.learning_rate
                    # self.u[0] = self.u[0] + target * c
            c = c + 1
        self.w[1:] = self.w[1:] - self.u[1:] / c
        # self.w[0] = np.array([self.w[0] - self.u[0] / c])
        print('The weight vector is %s' %self.w)


    # def net(self, X):
    #     return np.dot(X, self.w[1:]) + self.w[0]

    # def perc_predict(self, X):
    #     return np.where(self.net(X) >= 0, 1, -1)

    def perc_pred_err(self, X, y):
        count = 0

        for xi, target in zip(X, y):
            if np.sign((np.dot(self.w[1:], xi) + self.w[0])) != target:
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
    for i in range(1,11):
        print('T=%s' %i)
        pr = Average_Perceptron(0.1,i)
        pr.perc_fit(trainX, trainY)
        pr.perc_pred_err(testX,testY)




    # def perc_fit(self, X, y):
    #     X_len = len(X)
    #     randList = []
    #     self.w = np.zeros(1 + X.shape[1], dtype=float)
    #     A = np.zeros(len(self.w[1:]))
    #     B = 0
    #     for i in range(self.T):
    #         randList += np.random.permutation(X_len).tolist()
    #     for i in range(self.T * X_len):
    #         # print(y[randList[i]])X[randList[i]][-1]
    #         if (y[randList[i]]) * (np.inner(self.w[1:], X[randList[i]][0:len(self.w)]) +self.w[0]) <= 0:
    #             # print(self.w[1:])
    #             for j in X[randList[i]][0:len(self.w)]:
    #                 self.w[1:] += self.learning_rate * y[randList[i]] * j
    #             # print(i)
    #             self.w[0] += self.learning_rate * y[randList[i]]
    #             A = A + self.w[1:]
    #             B = B + self.w[0]
    #     print(A,B)

    # w = w + [rate * (train_data[permu[i]][-1]) * x for x in train_data[permu[i]][0:Num_attr]]