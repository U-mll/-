'''
    @Project   :机器学习FATION.MINIST
    @FileName  :my_knn.py
    @Time      :2025/12/31-8:15
    @Author    :@马丽霞
'''

import numpy as np


class MyKNeighborsClassifier:
    """纯手动实现的KNN分类器（欧氏距离 + 多数投票，无任何sklearn依赖）"""
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors  # K值
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """KNN是惰性学习，fit仅保存训练数据，无训练计算"""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        """预测核心逻辑：遍历样本+计算欧式距离+取K近邻+投票"""
        X_test = np.array(X_test)
        predictions = []
        for x in X_test:
            # 1. 计算当前测试样本与所有训练样本的欧式距离
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # 2. 按距离排序，取前K个近邻的索引
            k_indices = distances.argsort()[:self.n_neighbors]
            # 3. 近邻标签投票
            k_labels = self.y_train[k_indices]
            pred = np.bincount(k_labels).argmax()
            predictions.append(pred)
        return np.array(predictions)