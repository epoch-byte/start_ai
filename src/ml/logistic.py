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
梯度下降算法
"""


def grad_descent(data_array, data_label_array):
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
随机梯度下降(SGD)
"""


def gradRandomDes(data_array, data_label_array):
    max_cycle = 400
    data_mat = np.mat(data_array)
    #  label_mat = np.mat(data_label_array).transpose()
    m, n = np.shape(data_mat)
    wights = np.ones((n, 1))
    errors = []
    for j in range(max_cycle):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            index = int(np.random.uniform(0, len(data_index)))
            y = sigmoid(np.sum(data_mat[data_index[index]] * wights))
            # 损失函数
            error = y - data_label_array[data_index[index]]
            # 此处计算梯度，涉及矩阵求导
            wights = wights - alpha * data_mat[data_index[index]].transpose() * error
            del (data_index[index])
            errors.append(error)
    return wights


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
    weights = grad_descent(data_arr, data_label)
    print(weights.getA())
    plot_best_fit(weights)
    print("OK")


def colicTrain():
    data_arr, label_arr = loadDataSet("D:\\pySpace\\github\\start_ai\\data\\ml\\Logistic\\HorseColicTraining.txt")
    weights = gradRandomDes(data_arr, label_arr).getA()
    print(weights)
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

    X_test, y_test = loadDataSet(
        "D:\\pySpace\\github\\start_ai\\data\\ml\\Logistic\\HorseColicTraining_100k_last_column_binary.txt")
    print('Accuracy of LR Classifier:%f' % lr.score(X_test, y_test))
    #   print(classification_report(y_test, lr_y_predit, target_names=['high', 'low']))
    m, n = np.shape(X_test)

    r = lr.predict(X_test)
    error = 0
    for i in range(m):
        if r[i] != y_test[i]:
            error += 1
    print(error / m)


def generate_random_data_with_binary_last_column(num_records):
    random_data = []
    for _ in range(num_records):
        # First 21 columns: random floats between 0 and 100
        row = [f"{np.random.uniform(0, 100):.6f}" for _ in range(21)]
        # Last column: either 0 or 1
        row.append(f"{np.random.choice([0, 1])}")
        random_data.append("\t".join(row))
    return random_data


def createData():
    # Generate 100,000 records with the last column as 0 or 1
    random_data_binary_last_column = generate_random_data_with_binary_last_column(10000)

    # Save the modified data to a new text file
    random_file_binary_last_column_path = 'D:\pySpace\github\start_ai\data\ml\Logistic\HorseColicTraining_100k_last_column_binary.txt'
    with open(random_file_binary_last_column_path, 'w') as output_file:
        output_file.write("\n".join(random_data_binary_last_column))


if __name__ == "__main__":
    # createData()
    # simpleTest();
    colicPredict()

    scikitLearnLR()
