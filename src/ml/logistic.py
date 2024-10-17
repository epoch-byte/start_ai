#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
 load data from file
 return: 两个数组 
    data_arr   特征数据
    data_label 标签数据
"""
from random import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def loadDataSet(file):
    data_arr = []
    data_label_arr = []

    f = open(file, 'r')
    for line in f.readlines():
        line_arr = line.strip().split()
        array = [1.0]
        for item in line_arr[0:-1]:
            array.append(float(item))
        data_arr.append(array)
        data_label_arr.append(float(line_arr[-1]))
    return data_arr, data_label_arr


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


"""
批量梯度下降算法（batch BSD）
"""


def batchGradDes(data_array, data_label_array):
    alpha = 0.002
    max_cycle = 300

    data_mat = np.mat(data_array)
    label_mat = np.mat(data_label_array).transpose()
    m, n = np.shape(data_mat)

    wights = np.ones((n, 1))
    for i in range(max_cycle):
        y = sigmoid(data_mat * wights)
        # 损失函数
        error = y - label_mat
        # 此处计算梯度，涉及矩阵求导
        wights = wights - alpha * data_mat.transpose() * error
    return wights


"""
量随机梯度下降（BGD）
"""


def gradRandomDes(data_array, data_label_array):
    max_cycle = 200
    data_mat = np.mat(data_array)
    #  label_mat = np.mat(data_label_array).transpose()
    m, n = np.shape(data_mat)
    wights = np.ones((n, 1))
    errors = []
    alpha = 0.0001
    for j in range(max_cycle):
        index = int(np.random.uniform(0, len(data_mat)))
        y = sigmoid(data_mat[index] * wights)
        # 损失函数
        error = y - data_label_array[index]
        # 此处计算梯度，涉及矩阵求导
        wights = wights - alpha * data_mat[index].transpose() * error
        errors.append(error)

    x_label = list(range(0, max_cycle))
    plt.plot(x_label, errors)
    plt.show()
    return wights


"""
小批量随机梯度下降（Mini-batch SGD）
"""


def gradSmallBatchRandomDes(data_array, data_label_array):
    batchSize = 50
    max_cycle = 1000
    alpha = 0.000859

    data_mat = np.mat(data_array)
    m, n = np.shape(data_mat)
    weights = np.ones((n, 1))
    errors = []
    for j in range(max_cycle):
        batch_data_array, batch_data_label_array = createBatch(data_array, data_label_array, batchSize)

        batch_data_mat = np.mat(batch_data_array)
        batch_label_mat = np.mat(batch_data_label_array).transpose()

        y = sigmoid(batch_data_mat * weights)
        # 损失函数
        error = y - batch_label_mat
        # 此处计算梯度，涉及矩阵求导
        weights = weights - alpha * batch_data_mat.transpose() * error
        errors.append(np.mean(error))

    x_label = list(range(0, max_cycle))
    plt.plot(x_label, errors)
    plt.show()
    return weights


def createBatch(data_array, data_label_array, batch_size):
    random_numbers = [np.random.randint(0, len(data_label_array)) for _ in range(batch_size)]
    batch_data_array = []
    batch_data_label_array = []
    for i in random_numbers:
        batch_data_array.append(data_array[i])
        batch_data_label_array.append(data_label_array[i])
    return batch_data_array, batch_data_label_array


"""
梯度上升算法
"""


def gradAscCent(data_array, data_label_array):
    alpha = 0.001
    max_cycle = 100

    data_mat = np.mat(data_array)
    label_mat = np.mat(data_label_array).transpose()
    m, n = np.shape(data_mat)

    weights = np.ones((n, 1))
    for i in range(max_cycle):
        y = sigmoid(data_mat * weights)
        # 收益函数
        error = label_mat - y
        # 此处计算梯度，涉及矩阵求导
        weights = weights + alpha * data_mat.transpose() * error
    return weights


def viewData(data_array, data_label_arr):
    data_mat = np.array(data_array)
    n = np.shape(data_mat)[0]

    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if data_label_arr[i] == 1:
            x_cord1.append(data_mat[i, 1])
            y_cord1.append(data_mat[i, 2])
        else:
            x_cord2.append(data_mat[i, 1])
            y_cord2.append(data_mat[i, 2])

    plt.scatter(x_cord1, y_cord1, color='k', marker='^')
    plt.scatter(x_cord2, y_cord2, color='red', marker='s')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_best_fit(weights):
    data_mat, label_mat = loadDataSet("D:\\pySpace\\github\\start_ai\\data\\ml\\Logistic\\TestSet.txt")
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
    ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')
    x = np.arange(-3.0, 3.0, 0.1)
    weightsArr = weights.getA()
    y = (-weightsArr[0] - weightsArr[1] * x) / weightsArr[2]

    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

    yn = classify_vector(data_mat, weights)
    viewData(data_mat, yn)


def classify_vector(data_mat, weights):
    yn = []
    y1 = sigmoid(data_mat * weights).getA()
    for i in y1:
        if i >= 0.5:
            yn.append(1)
        else:
            yn.append(0)
    return yn


def simpleTest():
    data_arr, data_label = loadDataSet("D:\\pySpace\\github\\start_ai\\data\\ml\\Logistic\\TestSet.txt")
    viewData(data_arr, data_label)
    print("OK")
    weights = batchGradDes(data_arr, data_label)
    print(weights.getA())
    plot_best_fit(weights)
    print("OK")


def colicTrain():
    data_arr, label_arr = loadDataSet("D:\\pySpace\\github\\start_ai\\data\\ml\\Logistic\\HorseColicTraining.txt")
    # weights = gradRandomDes(data_arr, label_arr).getA()
    # weights = gradSmallBatchRandomDes(data_arr,label_arr).getA()
    weights = gradSmallBatchRandomDes(data_arr, label_arr).getA()
    return weights


def colicTest(weights):
    data_arr, label_arr = loadDataSet("D:\\pySpace\\github\\start_ai\\data\\ml\\Logistic\\HorseColicTest.txt")
    data_mat = np.mat(data_arr)
    predict = classify_vector(data_mat, weights)
    error = 0
    for i in range(len(label_arr)):
        if predict[i] != label_arr[i]:
            error += 1
    return error / len(label_arr)


def colicPredict():
    weights = colicTrain()
    errorRate = colicTest(weights)
    print(errorRate)


def scikitLearnLR():
    lr = LogisticRegression()
    X_train, y_train = loadDataSet("D:\\pySpace\\github\\start_ai\\data\\ml\\Logistic\\HorseColicTraining.txt")
    lr.fit(X_train, y_train)

    X_test, y_test = loadDataSet("D:\\pySpace\\github\\start_ai\\data\\ml\\Logistic\\HorseColicTest.txt")
    print('Accuracy of LR Classifier:%f' % lr.score(X_test, y_test))
    #   print(classification_report(y_test, lr_y_predit, target_names=['high', 'low']))
    m, n = np.shape(X_test)

    r = lr.predict(X_test)
    error = 0
    for i in range(m):
        if r[i] != y_test[i]:
            error += 1
    print(error / m)


if __name__ == "__main__":
    # simpleTest();
    colicPredict()
    scikitLearnLR()
