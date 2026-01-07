'''
    @Project   :机器学习FATION.MINIST
    @FileName  :01-Sklearn库实现KNN分类.py
    @Time      :2025/12/27-8:11
    @Author    :@马丽霞
'''
"""
Fashion-MNIST分类项目 - KNN算法核心实现与对比
任务：重点实现KNN算法，使用随机森林作为性能参照
包含：数据加载、预处理、KNN模型构建、随机森林对比、可视化分析
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class FashionMNISTKNNClassifier:
    """Fashion-MNIST KNN分类器（核心算法对比）"""

    def __init__(self, data_path='./data/FashionMNIST/raw/'):
        """
        初始化分类器
        参数:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = None
        self.models = {}
        self.results = {}

        # 类别名称
        self.class_names = ['T恤/上衣', '裤子', '套衫', '裙子', '外套',
                            '凉鞋', '衬衫', '运动鞋', '包', '靴子']

        # 算法配置
        self.knn_configs = [
            {'name': 'KNN-k3', 'k': 3, 'algorithm': 'auto'},
            {'name': 'KNN-k5', 'k': 5, 'algorithm': 'auto'},
            {'name': 'KNN-k7', 'k': 7, 'algorithm': 'auto'},
            {'name': 'KNN-ball_tree', 'k': 5, 'algorithm': 'ball_tree'},
            {'name': 'KNN-kd_tree', 'k': 5, 'algorithm': 'kd_tree'},
            {'name': 'KNN-brute', 'k': 5, 'algorithm': 'brute'}
        ]

    def load_data(self, sample_size=10000):
        """加载Fashion-MNIST数据集"""
        print("1. 数据加载")

        def load_idx_file(images_path, labels_path):
            """加载IDX格式文件"""
            with open(images_path, 'rb') as f:
                magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
                images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)

            with open(labels_path, 'rb') as f:
                magic, num = struct.unpack('>II', f.read(8))
                labels = np.fromfile(f, dtype=np.uint8)

            return images, labels

        try:
            # 从本地文件加载
            X_train_full, y_train_full = load_idx_file(
                f'{self.data_path}train-images-idx3-ubyte',
                f'{self.data_path}train-labels-idx1-ubyte'
            )
            X_test_full, y_test_full = load_idx_file(
                f'{self.data_path}t10k-images-idx3-ubyte',
                f'{self.data_path}t10k-labels-idx1-ubyte'
            )
            print("从本地文件加载成功")

        except FileNotFoundError:
            # 如果本地文件不存在，从在线源加载
            print("本地文件未找到，尝试在线加载...")
            try:
                from sklearn.datasets import fetch_openml
                X, y = fetch_openml('Fashion-MNIST', version=1,
                                    return_X_y=True, as_frame=False, parser='auto')
                X_train_full, X_test_full = X[:60000], X[60000:]
                y_train_full, y_test_full = y[:60000].astype(int), y[60000:].astype(int)
                print("在线数据加载成功")
            except Exception as e:
                print(f"数据加载失败: {e}")
                return False

        # 使用样本子集加速训练
        if sample_size and sample_size < len(X_train_full):
            print(f"使用样本子集: {sample_size}个训练样本")
            np.random.seed(42)
            indices = np.random.choice(len(X_train_full), sample_size, replace=False)
            self.X_train = X_train_full[indices]
            self.y_train = y_train_full[indices]

            # 测试集也使用子集
            test_size = min(2000, len(X_test_full))
            test_indices = np.random.choice(len(X_test_full), test_size, replace=False)
            self.X_test = X_test_full[test_indices]
            self.y_test = y_test_full[test_indices]
        else:
            self.X_train, self.y_train = X_train_full, y_train_full
            self.X_test, self.y_test = X_test_full, y_test_full

        # 显示数据集信息
        self.display_dataset_info()
        return True

    def display_dataset_info(self):
        """显示数据集基本信息"""
        print(f"\n数据集基本信息:")
        print(f"训练集大小: {self.X_train.shape[0]} 个样本")
        print(f"测试集大小: {self.X_test.shape[0]} 个样本")
        print(f"特征维度: {self.X_train.shape[1]} (28x28像素)")
        print(f"类别数量: {len(np.unique(self.y_train))}")

        # 类别分布
        print("\n训练集类别分布:")
        for i in range(10):
            count = np.sum(self.y_train == i)
            print(f"  {self.class_names[i]:<8}: {count:>5} ({count / len(self.y_train) * 100:.1f}%)")

        # 可视化部分样本
        self.visualize_samples()

    def visualize_samples(self, n_samples=10):
        """可视化样本图像"""
        plt.figure(figsize=(12, 5))
        for i in range(n_samples):
            plt.subplot(2, 5, i + 1)
            plt.imshow(self.X_train[i].reshape(28, 28), cmap='gray')
            plt.title(f"类别: {self.class_names[self.y_train[i]]}", fontsize=9)
            plt.axis('off')
        plt.suptitle('Fashion-MNIST 数据集样本示例', fontsize=14)
        plt.tight_layout()
        plt.savefig('./result/knn数据集样本示例.png', dpi=300, bbox_inches='tight')
        plt.show()

    def preprocess_data(self):
        """数据预处理"""
        print("\n" + "=" * 60)
        print("2. 数据预处理")

        # 1. 归一化到[0, 1]
        print("步骤1: 像素值归一化 (0-1)")
        X_train_norm = self.X_train / 255.0
        X_test_norm = self.X_test / 255.0

        # 2. 标准化 (Z-score标准化)
        print("步骤2: 标准化 (Z-score)")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train_norm)
        self.X_test_scaled = self.scaler.transform(X_test_norm)

        # 可视化预处理效果
        self.visualize_preprocessing(X_train_norm[0])
        print("数据预处理完成")

    def visualize_preprocessing(self, sample_image):
        """可视化预处理效果"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 原始图像
        axes[0].imshow(sample_image.reshape(28, 28), cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        # 归一化后
        axes[1].imshow(sample_image.reshape(28, 28), cmap='gray')
        axes[1].set_title('归一化后图像')
        axes[1].axis('off')

        # 标准化后像素值分布
        if self.scaler is not None and hasattr(self, 'X_train_scaled'):
            scaled_sample = self.X_train_scaled[0]
            axes[2].hist(scaled_sample, bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[2].set_title('标准化后分布')
            axes[2].set_xlabel('像素值')
            axes[2].set_ylabel('频数')

        plt.suptitle('数据预处理效果对比', fontsize=14)
        plt.tight_layout()
        plt.savefig('./result/knn数据预处理效果对比.png', dpi=300, bbox_inches='tight')
        plt.show()

    def train_knn_models(self):
        """训练多个KNN模型（不同参数）"""
        print("\n" + "=" * 60)
        print("3. KNN模型训练与参数分析")
        print("训练不同参数配置的KNN模型...")

        for config in self.knn_configs:
            print(f"\n训练 {config['name']} (k={config['k']}, algorithm={config['algorithm']})...")
            start_time = time.time()

            # 创建KNN模型
            knn = KNeighborsClassifier(
                n_neighbors=config['k'],
                algorithm=config['algorithm'],
                n_jobs=-1
            )

            # 训练模型
            knn.fit(self.X_train_scaled, self.y_train)
            train_time = time.time() - start_time

            # 预测
            start_time = time.time()
            y_pred = knn.predict(self.X_test_scaled)
            predict_time = time.time() - start_time

            # 计算评估指标
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted'
            )

            # 保存模型和结果
            self.models[config['name']] = knn
            self.results[config['name']] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'train_time': train_time,
                'predict_time': predict_time,
                'y_pred': y_pred,
                'k': config['k'],
                'algorithm': config['algorithm']
            }

            print(f"训练时间: {train_time:.2f}s")
            print(f"预测时间: {predict_time:.2f}s")
            print(f"准确率: {accuracy:.4f}")

        print("\n所有KNN模型训练完成")

    def train_random_forest(self):
        """训练随机森林作为性能参照"""
        print("\n" + "=" * 60)
        print("4. 随机森林模型（性能参照）")
        print("训练随机森林模型作为性能参照...")
        start_time = time.time()

        # 创建随机森林模型
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

        # 训练模型
        rf.fit(self.X_train_scaled, self.y_train)
        train_time = time.time() - start_time

        # 预测
        start_time = time.time()
        y_pred = rf.predict(self.X_test_scaled)
        predict_time = time.time() - start_time

        # 计算评估指标
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )

        # 保存模型和结果
        self.models['RandomForest'] = rf
        self.results['RandomForest'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': train_time,
            'predict_time': predict_time,
            'y_pred': y_pred
        }

        print(f"训练时间: {train_time:.2f}s")
        print(f"预测时间: {predict_time:.2f}s")
        print(f"准确率: {accuracy:.4f}")

        # 特征重要性分析
        self.analyze_feature_importance(rf)

    def analyze_feature_importance(self, rf_model):
        """分析随机森林特征重要性"""
        importances = rf_model.feature_importances_

        # 找出最重要的像素位置
        top_n = 50
        indices = np.argsort(importances)[::-1][:top_n]

        # 可视化最重要的像素
        importance_map = np.zeros((28, 28))
        for idx in indices[:top_n]:
            row, col = divmod(idx, 28)
            importance_map[row, col] = importances[idx]

        plt.figure(figsize=(10, 8))
        plt.imshow(importance_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='特征重要性')
        plt.title(f'随机森林特征重要性热图（前{top_n}个最重要像素）', fontsize=14)
        plt.xlabel('列', fontsize=12)
        plt.ylabel('行', fontsize=12)
        plt.tight_layout()
        plt.savefig('./result/随机森林特征重要性热图.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_models(self):
        """评估模型性能"""
        print("\n" + "=" * 60)
        print("5. 模型性能评估与对比")

        # 打印性能对比
        print("\n模型性能对比:")
        print(
            f"{'模型':<20} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'训练时间(s)':<12} {'预测时间(s)':<12}")
        print("-" * 90)

        best_knn_model = None
        best_knn_accuracy = 0
        best_overall_model = None
        best_overall_accuracy = 0

        for name, result in self.results.items():
            print(f"{name:<20} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
                  f"{result['recall']:<10.4f} {result['f1']:<10.4f} "
                  f"{result['train_time']:<12.2f} {result['predict_time']:<12.2f}")

            if 'KNN' in name and result['accuracy'] > best_knn_accuracy:
                best_knn_accuracy = result['accuracy']
                best_knn_model = name

            if result['accuracy'] > best_overall_accuracy:
                best_overall_accuracy = result['accuracy']
                best_overall_model = name

        print(f"最佳KNN模型: {best_knn_model} (准确率: {best_knn_accuracy:.4f})")
        print(f"最佳整体模型: {best_overall_model} (准确率: {best_overall_accuracy:.4f})")

        # 可视化评估结果
        self.visualize_evaluation()

        # KNN参数影响分析
        self.analyze_knn_parameters()

        # 对最佳KNN模型进行详细分析
        if best_knn_model:
            self.detailed_knn_analysis(best_knn_model)

    def visualize_evaluation(self):
        """可视化评估结果"""
        # 1. 准确率对比柱状图
        model_names = [name for name in self.results.keys() if 'KNN' in name]
        model_names.append('RandomForest')

        accuracies = [self.results[name]['accuracy'] for name in model_names]
        predict_times = [self.results[name]['predict_time'] for name in model_names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 准确率对比
        colors = ['blue'] * (len(model_names) - 1) + ['red']
        bars = axes[0].bar(range(len(model_names)), accuracies, color=colors, alpha=0.7)
        axes[0].set_title('模型准确率对比', fontsize=14)
        axes[0].set_xlabel('模型')
        axes[0].set_ylabel('准确率')
        axes[0].set_xticks(range(len(model_names)))
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].set_ylim([0.7, 1.0])
        axes[0].grid(axis='y', alpha=0.3)

        # 在柱子上添加数值
        for bar, acc in zip(bars, accuracies):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

        # 预测时间对比
        bars = axes[1].bar(range(len(model_names)), predict_times, color=colors, alpha=0.7)
        axes[1].set_title('模型预测时间对比', fontsize=14)
        axes[1].set_xlabel('模型')
        axes[1].set_ylabel('预测时间 (秒)')
        axes[1].set_xticks(range(len(model_names)))
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)

        # 在柱子上添加数值
        for bar, time_val in zip(bars, predict_times):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{time_val:.2f}s', ha='center', va='bottom', fontsize=9)

        plt.suptitle('KNN与随机森林性能对比', fontsize=16)
        plt.tight_layout()
        plt.savefig('./result/KNN与随机森林性能对比.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_knn_parameters(self):
        """分析KNN参数对性能的影响"""
        print("\n" + "=" * 60)
        print("KNN参数影响分析")

        # 分析k值对准确率的影响
        knn_models = {name: result for name, result in self.results.items()
                      if 'KNN' in name and 'k' in result}

        if len(knn_models) >= 3:  # 至少有3个不同k值的模型
            k_values = []
            accuracies = []
            predict_times = []

            for name, result in sorted(knn_models.items()):
                if 'k' in result:
                    k_values.append(result['k'])
                    accuracies.append(result['accuracy'])
                    predict_times.append(result['predict_time'])

            # 绘制k值影响图
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # k值对准确率的影响
            axes[0].plot(k_values, accuracies, 'o-', linewidth=2, markersize=8)
            axes[0].set_xlabel('k值')
            axes[0].set_ylabel('准确率')
            axes[0].set_title('k值对KNN准确率的影响')
            axes[0].grid(True, alpha=0.3)

            for k, acc in zip(k_values, accuracies):
                axes[0].text(k, acc + 0.001, f'{acc:.3f}', ha='center', va='bottom')

            # k值对预测时间的影响
            axes[1].plot(k_values, predict_times, 's-', linewidth=2, markersize=8, color='red')
            axes[1].set_xlabel('k值')
            axes[1].set_ylabel('预测时间 (秒)')
            axes[1].set_title('k值对KNN预测时间的影响')
            axes[1].grid(True, alpha=0.3)

            for k, time_val in zip(k_values, predict_times):
                axes[1].text(k, time_val + 0.1, f'{time_val:.2f}s', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig('./result/k值对KNN的影响.png', dpi=300, bbox_inches='tight')
            plt.show()

            print(f"最佳k值: {k_values[np.argmax(accuracies)]} (准确率: {max(accuracies):.4f})")

        # 分析不同算法的影响
        print("\nKNN不同算法实现对比:")
        print("-" * 60)
        algorithm_models = {name: result for name, result in self.results.items()
                            if 'KNN' in name and 'algorithm' in result and 'k' in result and result['k'] == 5}

        for name, result in algorithm_models.items():
            print(f"{name:<20}: 准确率={result['accuracy']:.4f}, "
                  f"预测时间={result['predict_time']:.2f}s, "
                  f"算法={result['algorithm']}")

    def detailed_knn_analysis(self, model_name):
        """对指定KNN模型进行详细分析"""
        if model_name not in self.models or model_name not in self.results:
            return

        print(f"详细分析: {model_name}")

        model = self.models[model_name]
        result = self.results[model_name]
        y_pred = result['y_pred']

        # 1. 分类报告
        print("\n1. 分类报告:")
        print(classification_report(self.y_test, y_pred,
                                    target_names=self.class_names,
                                    digits=4))

        # 2. 混淆矩阵
        print("\n2. 混淆矩阵:")
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{model_name} 混淆矩阵', fontsize=16)
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'./result/knn混淆矩阵.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. 各类别准确率
        print("\n3. 各类别准确率:")
        class_accuracies = []
        for i in range(10):
            idx = self.y_test == i
            if np.sum(idx) > 0:
                acc = accuracy_score(self.y_test[idx], y_pred[idx])
                class_accuracies.append(acc)
                print(f"  {self.class_names[i]:<8}: {acc:.4f}")

        # 可视化各类别准确率
        plt.figure(figsize=(12, 5))
        bars = plt.bar(range(10), class_accuracies, color='steelblue', alpha=0.8)
        plt.axhline(y=np.mean(class_accuracies), color='red', linestyle='--',
                    label=f'平均准确率: {np.mean(class_accuracies):.3f}')
        plt.xlabel('类别', fontsize=12)
        plt.ylabel('准确率', fontsize=12)
        plt.title(f'{model_name} 各类别分类准确率', fontsize=14)
        plt.xticks(range(10), self.class_names, rotation=45, ha='right')
        plt.ylim([0, 1.1])
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)

        for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'./result/knn各类别分类准确率.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. 错误分析
        self.analyze_knn_errors(model_name, y_pred)

    def analyze_knn_errors(self, model_name, y_pred):
        """分析KNN模型的错误分类"""
        errors = self.y_test != y_pred
        error_count = np.sum(errors)

        if error_count > 0:
            print(f"\n4. 错误分析 (共{error_count}个错误，错误率: {error_count / len(self.y_test) * 100:.2f}%):")

            # 找出最常见的错误类型
            error_pairs = []
            for i in range(len(self.y_test)):
                if errors[i]:
                    error_pairs.append((self.y_test[i], y_pred[i]))

            from collections import Counter
            common_errors = Counter(error_pairs).most_common(5)

            print("  最常见错误类型:")
            for (true_label, pred_label), count in common_errors:
                print(f"    {self.class_names[true_label]} → {self.class_names[pred_label]}: {count}次")

            # 可视化一些错误样本
            error_indices = np.where(errors)[0][:5]

            if len(error_indices) > 0:
                plt.figure(figsize=(12, 4))
                for i, idx in enumerate(error_indices[:5]):
                    plt.subplot(1, 5, i + 1)
                    plt.imshow(self.X_test[idx].reshape(28, 28), cmap='gray')
                    plt.title(f"真实: {self.class_names[self.y_test[idx]]}\n预测: {self.class_names[y_pred[idx]]}",
                              fontsize=9)
                    plt.axis('off')
                plt.suptitle(f'{model_name} 错误分类示例', fontsize=14)
                plt.tight_layout()
                plt.savefig('./result/knn错误分类示例.png', dpi=300, bbox_inches='tight')
                plt.show()

        # 显示生成的文件
        print("\n生成的文件:")
        files = [
            "fashion_samples_knn.png - 数据集样本可视化",
            "preprocessing_knn.png - 数据预处理效果",
            "knn_vs_rf_comparison.png - 模型性能对比",
            "knn_k_parameter_analysis.png - KNN参数分析",
            "knn_confusion_matrix.png - KNN混淆矩阵",
            "knn_class_accuracy.png - KNN各类别准确率",
            "knn_error_examples.png - KNN错误示例",
            "rf_feature_importance.png - 随机森林特征重要性"
        ]

        for file_info in files:
            print(f"  ✓ {file_info}")

    def run_pipeline(self, sample_size=10000):
        """运行完整的分类流程"""
        print("Fashion-MNIST分类项目 - KNN算法核心实现与对比")
        print("项目目标: 重点研究KNN算法，以随机森林为性能参照")

        # 1. 数据加载
        if not self.load_data(sample_size):
            return

        # 2. 数据预处理
        self.preprocess_data()

        # 3. 训练KNN模型
        self.train_knn_models()

        # 4. 训练随机森林（性能参照）
        self.train_random_forest()

        # 5. 模型评估
        self.evaluate_models()


def main():
    """主函数"""
    # 创建分类器实例
    classifier = FashionMNISTKNNClassifier()

    # 运行完整流程
    classifier.run_pipeline(sample_size=10000)  # 使用10000个样本进行实验


if __name__ == "__main__":
    main()