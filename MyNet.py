import pickle
import numpy as np
from MyFunc import *


class Net:
    def __init__(self, net_info, seed=0):
        np.random.seed(seed)

        self.net = []
        for layer in net_info:
            in_features = layer["in_features"]
            out_features = layer["out_features"]
            activation = layer["activation"]
            W = np.random.randn(out_features, in_features) * 0.01
            b = np.random.randn(out_features) * 0.01
            self.net.append((W, b, activation()))

    def load(self, path):
        with open(path, "rb") as f:
            self.net = pickle.load(f)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.net, f)

    # X: (batch_size, in_features)
    # W: (out_features, in_features)
    # b: (out_features)
    # U: (batch_size, out_features)
    def _forward(self, X):
        outputs = [{"U": None, "X": X}]
        for layer in self.net:
            W, b, f = layer
            U = np.dot(X, W.T) + b
            X = f.forward(U)
            outputs.append({"U": U, "X": X})
        return outputs

    # 将 dL/dU1 简记为 dU1_
    # X1=f(U1), U1=W0*X0+b0
    # dL/dU1 = dL/dX1 * dX1/dU1
    # dL/dW0 = dL/dU1 * dU1/dW0
    # dL/db0 = dL/dU1 * dU1/db0
    def _backward(self, outputs, dL_dX, lr):
        for i in range(len(self.net) - 1, -1, -1):
            # W: (out, in)
            # b: (out)
            W, b, f = self.net[i]

            # X0: (batch, in)
            # U1: (batch, out)
            X0 = outputs[i]["X"]
            U1 = outputs[i + 1]["U"]

            # 逐元素相乘
            # dL/dX1:  (batch, out)
            # dX1/dU1: (batch, out)
            # dL/dU1:  (batch, out)
            # dL/dU1 = dL/dX1 * dX1/dU1
            dL_dU1 = f.backward(U1, dL_dX)

            batch_size = X0.shape[0]

            # dL/dU1: (batch, out)
            # dU1/dW: (batch, in)
            # dL/dW:  (out, in)
            # dL/dW = dL/dU1 * dU1/dW
            #       = dL/dU1 * X
            dL_dW = np.dot(dL_dU1.T, X0) / batch_size

            # dL/dU1: (batch, out)
            # dU1/db: (out)
            # dL/db:  (out)
            # dL/db = dL/dU1 * dU1/db
            #       = dL/dU1
            dL_db = np.sum(dL_dU1, axis=0) / batch_size

            # dL/dU1: (batch, out)
            # dU1/dX: (out, in)
            # dL/dX:  (batch, in)
            # dL/dX = dL/dU1 * dU1/dX
            #       = dL/dU1 * W
            dL_dX = np.dot(dL_dU1, W)

            # 更新参数
            W -= lr * dL_dW
            b -= lr * dL_db

    def train(self, X, Y, lossF, batch_size, lr):
        for i in range(0, len(X), batch_size):
            X_batch = X[i : i + batch_size]
            Y_batch = Y[i : i + batch_size]
            outputs = self._forward(X_batch)

            dX = lossF.backward(Y_batch, outputs[-1]["X"])
            self._backward(outputs, dX, lr)

    def pred(self, X):
        return self._forward(X)[-1]["X"]
