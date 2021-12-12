from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd


def transfer_(X, y):
    data = np.zeros((len(X), 6))
    for i in range(len(X)):
        data[i, 4] = 1
        data[i, 5] = y[i]
        data[i, 0:4] = X[i]
    return data


df1 = pd.read_csv('train.csv', header=None)
trainY = df1.iloc[:, -1].values

trainX = df1.loc[:, 0:df1.shape[1] - 2].values
df2 = pd.read_csv('test.csv', header=None)
testX = df2.loc[:, 0:df2.shape[1] - 2].values
testY = df2.iloc[:, -1].values

traindata = transfer_(trainX, trainY)
testdata = transfer_(testX, testY)

trainY = np.array([row[-1] for row in traindata])
testY = np.array([row[-1] for row in testdata])
trainX = [np.array(row[0:5], ndmin=2) for row in traindata]
testX = [np.array(row[0:5], ndmin=2) for row in testdata]
train_data_array = np.array(trainX)
test_data_array = np.array(testX)
# print('train data dimension:', train_data_array.shape)
# print('test data dimension:', test_data_array.shape)

width_ = [5, 10, 25, 50, 100]
# width = 5
# width = 10
# width = 25
# width = 50
for width in width_:
    ########===========================three layer & relu
    model = keras.Sequential([keras.layers.Flatten(input_shape=(1, 5)),
                              keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
                                                 bias_initializer='zeros'),
                              keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
                                                 bias_initializer='zeros'),
                              keras.layers.Dense(2, activation=tf.nn.softmax)])

    ########===========================three layer & tanh
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(1,5)),
    #     keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                 bias_initializer='zeros'),
    #     keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                 bias_initializer='zeros'),
    #     keras.layers.Dense(2, activation=tf.nn.softmax)
    # ])

    ########===========================five layer & relu===========================
    # model = keras.Sequential([keras.layers.Flatten(input_shape=(1, 5)),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(2, activation=tf.nn.softmax)])

    ########===========================five layer & tanh
    # model = keras.Sequential([keras.layers.Flatten(input_shape=(1, 5)),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(2, activation=tf.nn.softmax)])

    ########===========================nine layer & relu===========================
    # model = keras.Sequential([keras.layers.Flatten(input_shape=(1, 5)),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(2, activation=tf.nn.softmax)])

    ########===========================nine layer & tanh
    # model = keras.Sequential([keras.layers.Flatten(input_shape=(1, 5)),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal',
    #                                              bias_initializer='zeros'),
    #                           keras.layers.Dense(2, activation=tf.nn.softmax)])

    print('\n ###### width is %s ######'%width)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data_array, trainY, epochs=20)

    test_loss, test_acc = model.evaluate(test_data_array, testY)
