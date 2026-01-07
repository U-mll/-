'''
    @Project   :机器学习FATION.MINIST
    @FileName  :02-自编算法实现KNN分类.py
    @Time      :2025/12/31-9:02
    @Author    :@马丽霞
'''

import numpy as np
import matplotlib.pyplot as plt
import struct
import time
import warnings
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 导入手动实现的KNN
from my_knn import MyKNeighborsClassifier

# 过滤无关警告
warnings.filterwarnings('ignore')
# 设置中文字体，解决中文乱码和负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

save_path = './result/'

# Fashion-MNIST 类别名称
class_names = ['T恤/上衣', '裤子', '套衫', '裙子', '外套',
               '凉鞋', '衬衫', '运动鞋', '包', '靴子']


def load_fashion_mnist(data_path='./data/FashionMNIST/raw/', sample_size=8000, test_size=2000):
    """加载Fashion-MNIST数据集【仅本地加载】，idx格式，适配原始数据集文件"""

    def load_idx_file(images_path, labels_path):
        # 读取图片文件
        with open(images_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
        # 读取标签文件
        with open(labels_path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return images, labels

    # 仅加载本地文件，无任何在线逻辑
    X_train_full, y_train_full = load_idx_file(f'{data_path}train-images-idx3-ubyte',
                                               f'{data_path}train-labels-idx1-ubyte')
    X_test_full, y_test_full = load_idx_file(f'{data_path}t10k-images-idx3-ubyte', f'{data_path}t10k-labels-idx1-ubyte')

    # 采样加速训练，保证两个模型公平对比
    np.random.seed(42)
    train_idx = np.random.choice(len(X_train_full), sample_size, replace=False)
    test_idx = np.random.choice(len(X_test_full), test_size, replace=False)
    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_test = X_test_full[test_idx]
    y_test = y_test_full[test_idx]

    # 数据预处理：归一化+标准化
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"数据集信息：训练样本={X_train.shape[0]} 测试样本={X_test.shape[0]} 特征维度={X_train.shape[1]}")
    return X_train, X_test, y_train, y_test


def model_train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """通用训练评估函数，统一计算耗时+准确率+预测结果"""
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred

    acc = accuracy_score(y_test, y_pred)
    print(f"\n【{model_name}】")
    print(f"训练耗时: {train_time:.4f} s")
    print(f"预测耗时: {pred_time:.4f} s")
    print(f"分类准确率: {acc:.4f}")
    print("分类报告：")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3))
    return acc, pred_time, train_time, y_pred


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """绘制混淆矩阵 + 自动保存高清图片"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_path}{filename}", dpi=300, bbox_inches='tight')
    plt.show()


def plot_compare_result(res_dict):
    """绘制【准确率+预测耗时】双对比图 + 自动保存，核心对比图"""
    model_names = list(res_dict.keys())
    accs = [res_dict[name]['acc'] for name in model_names]
    pred_times = [res_dict[name]['pred_time'] for name in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # 准确率对比
    bars1 = ax1.bar(model_names, accs, color=['forestgreen', 'royalblue'], alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('手动实现KNN VS Sklearn官方KNN 准确率对比', fontsize=14)
    ax1.set_ylabel('分类准确率')
    ax1.set_ylim([0.75, 0.90])
    ax1.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f'{acc:.4f}', ha='center', fontsize=12)

    # 预测耗时对比
    bars2 = ax2.bar(model_names, pred_times, color=['forestgreen', 'royalblue'], alpha=0.8, edgecolor='black',
                    linewidth=1)
    ax2.set_title('手动实现KNN VS Sklearn官方KNN 预测耗时对比', fontsize=14)
    ax2.set_ylabel('预测耗时 (秒)')
    ax2.grid(axis='y', alpha=0.3)
    for bar, t in zip(bars2, pred_times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{t:.2f}s', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{save_path}KNN对比_准确率+耗时.png", dpi=300, bbox_inches='tight')
    plt.show()


#  主函数
if __name__ == "__main__":
    print("Fashion-MNIST 分类任务 | 手动KNN VS Sklearn-KNN 对比 ")

    # 1. 加载本地数据集
    X_train, X_test, y_train, y_test = load_fashion_mnist(sample_size=8000, test_size=2000)

    # 2. 初始化两个对比模型 (K=5，公平对比)
    my_knn = MyKNeighborsClassifier(n_neighbors=5)  # 手动KNN
    sk_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)  # Sklearn KNN

    # 3. 训练评估
    result_dict = {}
    acc1, t_pred1, t_train1, y_pred1 = model_train_and_evaluate(my_knn, "手动实现KNN", X_train, X_test, y_train, y_test)
    result_dict["手动KNN"] = {"acc": acc1, "pred_time": t_pred1, "train_time": t_train1}

    acc2, t_pred2, t_train2, y_pred2 = model_train_and_evaluate(sk_knn, "Sklearn官方KNN", X_train, X_test, y_train,
                                                                y_test)
    result_dict["Sklearn-KNN"] = {"acc": acc2, "pred_time": t_pred2, "train_time": t_train2}

    # 4. 绘制并保存所有图片
    plot_confusion_matrix(y_test, y_pred1, "手动实现KNN 混淆矩阵", "手动KNN_混淆矩阵.png")
    plot_confusion_matrix(y_test, y_pred2, "Sklearn官方KNN 混淆矩阵", "SklearnKNN_混淆矩阵.png")
    plot_compare_result(result_dict)

    # 最终总结
    print(f"手动KNN  -> 准确率: {acc1:.4f} | 预测耗时: {t_pred1:.2f} s")
    print(f"Sklearn-KNN -> 准确率: {acc2:.4f} | 预测耗时: {t_pred2:.2f} s")