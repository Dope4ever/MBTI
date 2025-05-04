# 导入需要的库和模块
# 用于将数据集分割为训练集、验证集和测试集
from sklearn.model_selection import train_test_split, GridSearchCV

# 导入机器学习模型
# 导入逻辑回归模型
from sklearn.linear_model import LogisticRegression
# 导入支持向量机（SVM）模型
from sklearn.svm import SVC
# 导入高斯朴素贝叶斯模型
from sklearn.naive_bayes import GaussianNB
# 导入CatBoost模型
from catboost import CatBoostClassifier

# 导入评估指标
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 用于标签编码
from sklearn.preprocessing import LabelEncoder
import joblib  # 用于模型持久化，保存和加载模型
import numpy as np  # 用于数值计算
import json  # 用于处理JSON格式的数据

# 从FeatureEngine模块中导入特征工程处理函数
from FeatureEngine import advanced_feature_engineering
# 从LoadJSON模块中导入数据加载函数
from LoadJSON import load_data

# 定义MBTI分类的四个维度
dimensions = ['E/I', 'S/N', 'T/F', 'J/P']

# 定义超参数网格
param_grid_svm = {
    'C': [0.1, 1, 10, 100],  # 正则化参数C的取值范围
    'gamma': [0.001, 0.01, 0.1, 1],  # 核函数的gamma值
    'kernel': ['rbf']  # 选择RBF核函数
}

param_grid_lr = {
    'C': [0.1, 1, 10, 100],  # 正则化参数C的取值范围
    'solver': ['liblinear', 'saga'],  # 解算器
    'max_iter': [2000, 5000]  # 增加最大迭代次数
}

param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7]  # 高斯朴素贝叶斯的平滑参数
}

param_grid_catboost = {
    'iterations': [500, 1000],  # 迭代次数
    'learning_rate': [0.01, 0.05, 0.1],  # 学习率
    'depth': [6, 8, 10],  # 树的深度
    'loss_function': ['Logloss', 'CrossEntropy']  # 损失函数
}


# 训练和评估模型的函数
def train_and_evaluate(df, X, feature_objects):
    # 保存特征工程处理对象，以便后续使用
    joblib.dump(feature_objects, 'model/feature_processing_objects.pkl')

    # 遍历每个MBTI维度，训练对应的模型
    for dim in dimensions:
        print(f"\n{'=' * 40}\nTraining models for dimension: {dim}\n{'=' * 40}")

        # 准备标签数据，LabelEncoder用于将标签转换为数字
        le = LabelEncoder()
        y = le.fit_transform(df[dim])  # 将每个维度的标签从字符串转换为数字

        # 数据集划分：80%用于训练，10%用于验证，10%用于测试
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42)  # 首先划分出训练集和临时集（验证+测试集）

        # 从临时集再划分出10%作为验证集，剩下的10%作为测试集
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        best_model = None  # 存储表现最好的模型
        best_metrics = {'accuracy': 0, 'f1': 0}  # 初始化最佳模型的评估指标
        best_model_name = ''  # 最佳模型名称

        # 进行网格搜索，寻找最佳的SVM超参数
        print(f"Performing Grid Search for SVM...")

        svm = SVC(random_state=42, probability=True)
        grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=1)
        grid_search_svm.fit(X_train, y_train)
        best_svm_model = grid_search_svm.best_estimator_

        y_pred_svm = best_svm_model.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        precision_svm = precision_score(y_test, y_pred_svm)
        recall_svm = recall_score(y_test, y_pred_svm)
        f1_svm = f1_score(y_test, y_pred_svm)

        print(f"SVM Model after Grid Search:")
        print(
            f"Accuracy: {accuracy_svm:.4f}, Precision: {precision_svm:.4f}, Recall: {recall_svm:.4f}, F1-Score: {f1_svm:.4f}")

        if f1_svm > best_metrics['f1']:
            best_metrics = {'accuracy': accuracy_svm, 'f1': f1_svm, 'precision': precision_svm, 'recall': recall_svm}
            best_model = best_svm_model
            best_model_name = "SVM (Grid Search)"

        # 逻辑回归网格搜索
        print(f"Performing Grid Search for Logistic Regression...")

        lr = LogisticRegression(random_state=42, max_iter=5000)
        grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='accuracy', n_jobs=1)
        grid_search_lr.fit(X_train, y_train)
        best_lr_model = grid_search_lr.best_estimator_

        y_pred_lr = best_lr_model.predict(X_test)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        precision_lr = precision_score(y_test, y_pred_lr)
        recall_lr = recall_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)

        print(f"Logistic Regression Model after Grid Search:")
        print(
            f"Accuracy: {accuracy_lr:.4f}, Precision: {precision_lr:.4f}, Recall: {recall_lr:.4f}, F1-Score: {f1_lr:.4f}")

        if f1_lr > best_metrics['f1']:
            best_metrics = {'accuracy': accuracy_lr, 'f1': f1_lr, 'precision': precision_lr, 'recall': recall_lr}
            best_model = best_lr_model
            best_model_name = "Logistic Regression (Grid Search)"

        # 高斯朴素贝叶斯网格搜索
        print(f"Performing Grid Search for Gaussian Naive Bayes...")

        nb = GaussianNB()
        grid_search_nb = GridSearchCV(nb, param_grid_nb, cv=5, scoring='accuracy', n_jobs=1)
        grid_search_nb.fit(X_train.toarray(), y_train)  # Convert to dense array before fitting
        best_nb_model = grid_search_nb.best_estimator_

        y_pred_nb = best_nb_model.predict(X_test.toarray())  # Convert to dense array before predicting
        accuracy_nb = accuracy_score(y_test, y_pred_nb)
        precision_nb = precision_score(y_test, y_pred_nb)
        recall_nb = recall_score(y_test, y_pred_nb)
        f1_nb = f1_score(y_test, y_pred_nb)

        print(f"Gaussian Naive Bayes Model after Grid Search:")
        print(
            f"Accuracy: {accuracy_nb:.4f}, Precision: {precision_nb:.4f}, Recall: {recall_nb:.4f}, F1-Score: {f1_nb:.4f}")

        if f1_nb > best_metrics['f1']:
            best_metrics = {'accuracy': accuracy_nb, 'f1': f1_nb, 'precision': precision_nb, 'recall': recall_nb}
            best_model = best_nb_model
            best_model_name = "Gaussian Naive Bayes (Grid Search)"

        # CatBoost网格搜索
        print(f"Performing Grid Search for CatBoost...")

        catboost = CatBoostClassifier(random_seed=42, verbose=False)
        grid_search_catboost = GridSearchCV(catboost, param_grid_catboost, cv=5, scoring='accuracy', n_jobs=1)
        grid_search_catboost.fit(X_train, y_train)
        best_catboost_model = grid_search_catboost.best_estimator_

        y_pred_catboost = best_catboost_model.predict(X_test)
        accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
        precision_catboost = precision_score(y_test, y_pred_catboost)
        recall_catboost = recall_score(y_test, y_pred_catboost)
        f1_catboost = f1_score(y_test, y_pred_catboost)

        print(f"CatBoost Model after Grid Search:")
        print(
            f"Accuracy: {accuracy_catboost:.4f}, Precision: {precision_catboost:.4f}, Recall: {recall_catboost:.4f}, F1-Score: {f1_catboost:.4f}")

        if f1_catboost > best_metrics['f1']:
            best_metrics = {'accuracy': accuracy_catboost, 'f1': f1_catboost, 'precision': precision_catboost,
                            'recall': recall_catboost}
            best_model = best_catboost_model
            best_model_name = "CatBoost (Grid Search)"

        # 保存表现最好的模型
        if best_model:
            print(f"\n♦♦♦Best model for {dim}: {best_model_name}♦♦♦")
            joblib.dump(best_model, f'model/best_model_{dim.replace("/", "_")}.pkl')  # 保存最佳模型


# 主程序入口
if __name__ == '__main__':
    # 加载数据
    df = load_data('mbtibench.jsonl')  # 从jsonl文件中加载数据
    # 进行特征工程处理
    X, feature_objects = advanced_feature_engineering(df)
    # 保存特征工程处理对象
    joblib.dump(feature_objects, 'model/feature_processing_objects.pkl')
    # 训练并评估模型
    train_and_evaluate(df, X, feature_objects)
