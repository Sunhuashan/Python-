# 从零实现线性回归
import torch
import random
from d2l import torch as d2l


# 生成数据集 y = 2 * x1 - 3.4 * x2 + 4.2 + 噪声
def synthetic_data(w, b, example_num):
    X = torch.normal(0, 1, (example_num, len(w)))  # 生成样本，每个元素取自均值为0，标准差为1正态分布
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 1, y.size())  # 生成特征
    return X, y.reshape(-1, 1)


w_true = torch.tensor([2, -3.4])
b_true = 4.2
features, labels = synthetic_data(w_true, b_true, 1000)


# 分批获取数据集
def data_itor(batch_size, features, labels):
    example_num = len(features)
    indices = list(range(example_num))  # [0:n]
    random.shuffle(indices)  # 乱序
    for i in range(0, example_num, batch_size):
        batch_indices = indices[i:min(i + batch_size, example_num)]
        yield features[batch_indices], labels[batch_indices]


# 初始化参数
w = torch.normal(0, 0.01, (2, 1), requires_grad=True)
b = torch.zeros((1, 1), requires_grad=True)


# 定义损失函数
def loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # 平方损失函数


# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b  # 线性回归模型
