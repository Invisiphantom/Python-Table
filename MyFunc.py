import numpy as np
from abc import ABC, abstractmethod


# 激活函数 抽象类
# Y=f(U), U=W*X+b
# dL/dU = dL/dY * dY/dU
class ActFunc(ABC):
    @abstractmethod
    def forward(U):
        pass

    @abstractmethod
    def backward(U, dL_dY):
        pass


class Id(ActFunc):
    @staticmethod
    def forward(U):
        return U

    @staticmethod
    def backward(U, dL_dY):
        return dL_dY * 1


class Sigmoid(ActFunc):
    @staticmethod
    def forward(U):
        return 1 / (1 + np.exp(-U))

    @staticmethod
    def backward(U, dL_dY):
        X = Sigmoid.forward(U)
        return dL_dY * (X * (1 - X))


class Relu(ActFunc):
    @staticmethod
    def forward(U):
        return np.maximum(0, U)

    @staticmethod
    def backward(U, dL_dY):
        return dL_dY * np.where(U > 0, 1, 0)


class Gelu(ActFunc):
    @staticmethod
    def forward(U):
        return 0.5 * U * (1 + np.tanh(np.sqrt(2 / np.pi) * (U + 0.044715 * U**3)))

    @staticmethod
    def backward(U, dL_dY):
        return dL_dY * (0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (U + 0.044715 * U**3))) + 0.5 * U * (1 - np.tanh(np.sqrt(2 / np.pi) * (U + 0.044715 * U**3)) ** 2) * np.sqrt(2 / np.pi) * (1 + 0.134145 * U**2))


class Softmax(ActFunc):
    # (batch_size, out_features)
    @staticmethod
    def forward(U):
        exp_U = np.exp(U - np.max(U, axis=-1, keepdims=True))
        return exp_U / np.sum(exp_U, axis=-1, keepdims=True)

    @staticmethod
    def backward(U, dL_dY):
        X = Softmax.forward(U)
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
        return (_Y - Y) / len(Y)


class CrossEntropy(LossFunc):
    # Y : 真实标签 (one-hot)
    # _Y: 预测概率 (softmax)
    @staticmethod
    def forward(Y, _Y):
        epsilon = 1e-12
        _Y = np.clip(_Y, epsilon, 1 - epsilon)
        return -np.sum(Y * np.log(_Y)) / len(Y)

    @staticmethod
    def backward(Y, _Y):
        epsilon = 1e-12
        _Y = np.clip(_Y, epsilon, 1 - epsilon)
        return (-Y / _Y) / len(Y)
