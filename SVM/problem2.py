import numpy as np
import pandas as pd
import math
import statistics
import matplotlib.pyplot as plot
from numpy.linalg import inv

gamma_0 = 0.01
d = 1


class SVM_(object):
    def __init__(self, C, epochs, sch_flag):
        self.C = C
        self.w = None
        self.epochs = epochs
        self.sch_flag = sch_flag
        self.learning_rate = gamma_0
        pass

    def random_1(self, X, y):
        permu = np.random.permutation((len(X)))
        X_random = X[permu]
        y_random = y[permu]
        return X_random, y_random

    def SVM_fit(self, X, y):
        n = len(X)
        w = np.zeros(1 + X.shape[1], dtype=float)

        for i in range(self.epochs):
            X, y = self.random_1(X, y)

            for t, row in enumerate(X):
                if self.sch_flag == 0 and t > 0:
                    self.learning_rate = gamma_0 / (1 + (gamma_0 / d) * t)

                if self.sch_flag == 1 and t > 0:
                    self.learning_rate = gamma_0 / (1 + t)

                if y[t] * np.dot(row, w[1:]) <= 1:
                    w[1:] = w[1:] - self.learning_rate * w[1:] + self.learning_rate * self.C * len(X) * y[t] * row
                else:
                    w[1:] = (1 - self.learning_rate) * w[1:]

        self.w = w

    def SVM_predict(self, X, y):
        pred = np.sign(np.dot(X, self.w[1:]) + self.w[0])
        errors = 0
        for i in range(len(X)):
            if pred[i] != y[i]:
                errors += 1
        return errors / len(X)


if __name__ == '__main__':
    df1 = pd.read_csv('train.csv', header=None)
    y = df1.iloc[:, -1].values
    trainY = 2 * y - 1
    trainX = df1.loc[:, 0:df1.shape[1] - 2].values
    df2 = pd.read_csv('test.csv', header=None)
    testX = df2.loc[:, 0:df2.shape[1] - 2].values
    y = df2.iloc[:, -1].values
    testY = 2 * y - 1
    C = [1 / 873, 10 / 873, 50 / 873, 100 / 873, 300 / 873, 500 / 873, 700 / 873]
    print("When Schedule of Learning Rate Schedule is 0\r\n")

    for cc in C:
        svm_ = SVM_(C=cc, epochs=100, sch_flag=0)
        svm_.SVM_fit(trainX, trainY)
        print('C: %s' % cc)
        print('Weight: %s' % svm_.w)
        print('Train error: %.2f%% ' % (svm_.SVM_predict(trainX, trainY) * 100))
        print('Test error: %.2f%% ' % (svm_.SVM_predict(testX, testY) * 100))
        print("\r\n")
    print("When Schedule of Learning Rate Schedule is 1\r\n")

    for cc in C:
        svm_ = SVM_(C=cc, epochs=100, sch_flag=1)
        svm_.SVM_fit(trainX, trainY)
        print('C: %s' % cc)
        print('Weight: %s' % svm_.w)
        print('Train error: %.2f%% ' % (svm_.SVM_predict(trainX, trainY) * 100))
        print('Test error: %.2f%% ' % (svm_.SVM_predict(testX, testY) * 100))
        print("\r\n")
