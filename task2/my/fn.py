import numpy as np
from abc import ABC, abstractmethod


# 激活函数 抽象类
# Y=f(U), U=W*X+b
# dL/dU = dL/dY * dY/dU
class ActFunc(ABC):
    @abstractmethod
    def forward(X):
        pass

    @abstractmethod
    def backward(X, dL_dY, lr):
        pass


class Sigmoid(ActFunc):
    def __str__(self):
        return "Sigmoid()"
    
    @staticmethod
    def forward(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def backward(X, dL_dY, lr):
        X = Sigmoid.forward(X)
        return dL_dY * (X * (1 - X))


class Relu(ActFunc):
    def __str__(self):
        return "Relu()"
    
    @staticmethod
    def forward(X):
        return np.maximum(0, X)

    @staticmethod
    def backward(X, dL_dY, lr):
        return dL_dY * np.where(X > 0, 1, 0)


class Gelu(ActFunc):
    def __str__(self):
        return "Gelu()"
    
    @staticmethod
    def forward(X):
        return 0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X**3)))

    @staticmethod
    def backward(X, dL_dY, lr):
        return dL_dY * (0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X**3))) + 0.5 * X * (1 - np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X**3)) ** 2) * np.sqrt(2 / np.pi) * (1 + 0.134145 * X**2))


class Softmax(ActFunc):
    def __str__(self):
        return "Softmax()"
    
    # (batch_size, out_features)
    @staticmethod
    def forward(X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    @staticmethod
    def backward(X, dL_dY, lr):
        X = Softmax.forward(X)
        grad_sum = np.sum(X * dL_dY, axis=-1, keepdims=True)
        return X * (dL_dY - grad_sum)


# ------------------------


# 损失函数 抽象类
class LossFunc(ABC):
    @abstractmethod
    def forward(Y, _Y):
        pass

    @abstractmethod
    def backward(Y, _Y):
        pass


class Mse(LossFunc):
    @staticmethod
    def forward(Y, _Y):
        return np.mean((_Y - Y) ** 2)

    @staticmethod
    def backward(Y, _Y):
        return 2 * (_Y - Y) / Y.shape[0]


class CrossEntropy(LossFunc):
    # Y : 真实标签 (one-hot)
    # _Y: 预测概率 (softmax)
    @staticmethod
    def forward(Y, _Y):
        epsilon = 1e-12
        _Y = np.clip(_Y, epsilon, 1 - epsilon)
        return -np.sum(Y * np.log(_Y)) / Y.shape[0]

    @staticmethod
    def backward(Y, _Y):
        epsilon = 1e-12
        _Y = np.clip(_Y, epsilon, 1 - epsilon)
        return (-Y / _Y) / Y.shape[0]
