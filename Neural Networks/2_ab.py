import pandas as pd
import numpy as np


def transfer_(X, y):
    data = np.zeros((len(X), 6))
    for i in range(len(X)):
        data[i, 4] = 1
        data[i, 5] = y[i]
        data[i, 0:4] = X[i]
    return data

def gamma(t, gamma_0, d):
    return gamma_0 / (1 + (gamma_0 / d) * t)


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


class Network:
    def __init__(self,
                 input_nodes,
                 output_nodes,
                 hidden1_nodes,
                 hidden2_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden1_nodes = hidden1_nodes
        self.hidden2_nodes = hidden2_nodes
        self.parameter_ini()
        self.stepsize = gamma

    def parameter_ini(self):
        self.w_hidden1 = np.random.normal(0, 1, (self.hidden1_nodes - 1, self.input_nodes)).astype(np.float64)
        self.w_hidden2 = np.random.normal(0, 1, (self.hidden2_nodes - 1, self.hidden1_nodes)).astype(np.float64)
        self.w_output = np.random.normal(0, 1, (self.output_nodes, self.hidden2_nodes)).astype(np.float64)

    def forward(self, X):
        X = np.array(X, ndmin=2)
        z_hidden1 = sigmoid(np.dot(self.w_hidden1, X.T))
        z_hidden2 = sigmoid(np.dot(self.w_hidden2, np.concatenate((z_hidden1, [[1]]), axis=0)))
        output = np.dot(self.w_output, np.concatenate((z_hidden2, [[1]]), axis=0))
        return np.sign(output)

    def training(self, X, y, iter_, gamma0, d):
        X = np.array(X, ndmin=2)
        z_hidden1 = sigmoid(np.dot(self.w_hidden1, X.T))
        z_hidden2 = sigmoid(np.dot(self.w_hidden2, np.concatenate((z_hidden1, [[1]]), axis=0)))
        output = np.dot(self.w_output, np.concatenate((z_hidden2, [[1]]), axis=0))
        error_output = output - y
        grad_w_output = error_output * (np.concatenate((z_hidden2, [[1]]), axis=0)).T
        error_hidden2 = error_output * self.w_output[0, :][:-1]
        var_middle1 = error_hidden2 * z_hidden2.T * (1 - z_hidden2.T)
        var_middle2 = np.concatenate((z_hidden1, [[1]]), axis=0)
        grad_w_hidden2 = np.dot(var_middle2, var_middle1).T

        var_middle3 = self.w_output[0, :][:-1] * (z_hidden2.T * (1 - z_hidden2.T))
        var_middle4 = np.zeros((self.hidden1_nodes - 1, 1))
        var_middle5 = z_hidden1.T * (1 - z_hidden1.T)

        for i in range(self.hidden1_nodes - 1):
            var_middle4[i, 0] = error_output * np.inner(var_middle3, self.w_hidden2[:, i].T) * var_middle5[0, i]
        grad_w_hidden1 = np.dot(var_middle4, X)

        self.w_output = self.w_output - gamma(iter_, gamma0, d) * grad_w_output
        self.w_hidden2 = self.w_hidden2 - gamma(iter_, gamma0, d) * grad_w_hidden2
        self.w_hidden1 = self.w_hidden1 - gamma(iter_, gamma0, d) * grad_w_hidden1
        iter_ = iter_ + 1
        return iter_

if __name__ == "__main__":

    df1 = pd.read_csv('train.csv', header=None)
    y = df1.iloc[:, -1].values
    trainY = 2 * y - 1
    trainX = df1.loc[:, 0:df1.shape[1] - 2].values
    df2 = pd.read_csv('test.csv', header=None)
    testX = df2.loc[:, 0:df2.shape[1] - 2].values
    y = df2.iloc[:, -1].values
    testY = 2 * y - 1
    traindata = transfer_(trainX, trainY)
    testdata = transfer_(testX, testY)

    nw = Network(input_nodes=5, output_nodes=1, hidden1_nodes=100, hidden2_nodes=100)
    count_ = 0
    for i in range(len(trainX)):
        iter_ = nw.training(traindata[i, 0:5], traindata[i][-1], 1, 0.01, 2)
        y_pred = nw.forward(traindata[i, 0:5])
        if y_pred != trainY[i]:
            count_ = count_ + 1
        print(nw.w_hidden1)
    print('train error = ', count_ / len(trainX))

    for i in range(len(testX)):
        y_pred = nw.forward(testdata[i, 0:5])
        if y_pred != testY[i]:
            count_ = count_ + 1
    print('test error = ', count_ / len(testX))