#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Desc    : 一系列的评估函数f1, recall, acc, precision, confusion_matrix...
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import confusion_matrix


class Matrix:
    def __init__(self, y_true, y_pre, multi=False):
        self.true = y_true
        self.pre = y_pre
        # 是否是多分类, 默认二分类
        self.multi = multi  # average的参数有micro、macro、weighted,如果选择micro,那么recall和pre和acc没区别，建议使用macro，同时数据集最好已经没有不平衡的问题

    def get_acc(self):
        return accuracy_score(self.true, self.pre)

    def get_recall(self):
        # tp / (tp + fn)
        if self.multi:
            return recall_score(self.true, self.pre, average="macro")
        return recall_score(self.true, self.pre)

    def get_precision(self):
        # tp / (tp + fp)
        if self.multi:
            return precision_score(self.true, self.pre, average="macro")
        return precision_score(self.true, self.pre)

    def get_f1(self):
        # F1 = 2 * (precision * recall) / (precision + recall)
        if self.multi:
            return f1_score(self.true, self.pre, average="macro")
        return f1_score(self.true, self.pre)

    def get_confusion_matrix(self):
        return confusion_matrix(self.true, self.pre)


if __name__ == "__main__":
    # dic_labels = {0: 'W', 1: 'LS', 2: 'SWS', 3: 'REM', 4: 'E'}
    # cm = np.array([(193, 31, 0, 41, 42), (87, 1038, 32, 126, 125),
    #               (17, 337, 862, 1, 2), (17, 70, 0, 638, 54), (1, 2, 3, 4, 5)])
    # matrix_execute = Matrix(None, None)
    # matrix_execute.plot_confusion_matrix(cm, dic_labels)
    y_true = np.array([0] * 30 + [1] * 240 + [2] * 30)
    y_pred = np.array(
        [0] * 10
        + [1] * 10
        + [2] * 10
        + [0] * 40
        + [1] * 160
        + [2] * 40
        + [0] * 5
        + [1] * 5
        + [2] * 20
    )
    dic_labels = {0: 0, 1: 1, 2: 2}
    matrix_execute = Matrix(y_true=y_true, y_pre=y_pred, multi=True)
    print(matrix_execute.get_acc())
    print(matrix_execute.get_precision())
    print(matrix_execute.get_recall())
    print(matrix_execute.get_f1())
