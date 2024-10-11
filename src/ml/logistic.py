#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
 load data from file
 return: 两个数组 
    data_arr   特征数据
    data_label 标签数据
"""
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    data_arr = []
    data_label_arr = []

    f = open("D:\\pySpace\\github\\start_ai\\data\\ml\\Logistic\\TestSet.txt", 'r')
    for line in f.readlines():
        line_arr = line.strip().split()
        data_arr.append([1.0, float(line_arr[0]), float(line_arr[1])])
        data_label_arr.append(int(line_arr[2]))
    return data_arr, data_label_arr

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def grad_descent(data_arr, data_label_arr):
    batchSize = all
    alpha = 0.001
    max_cycle = 500

    data_mat = np.mat(data_arr)
    label_mat = np.mat(data_label_arr).transpose()
    m,n = np.shape(data_mat)

    wights = np.ones((n,1))

    for i in range(max_cycle):
        y = sigmoid(data_mat*wights)
        error = label_mat - y
        wights = wights - alpha
    return wights

def viewData(data_arr, data_label_arr):
    data_mat = np.array(data_arr)
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

    plt.scatter(x_cord1, y_cord1,  color='k', marker='^')
    plt.scatter(x_cord2, y_cord2,  color='red', marker='s')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
    data_arr, data_label = loadDataSet()
    viewData(data_arr, data_label)
    print("OK")
    wights = grad_ascent(data_arr,data_label)