import numpy as np
from MyFunc import *


class Mlp:
    def __init__(self, in_features, out_features, activation):
        self.in_features = in_features
        self.out_features = out_features

        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.random.randn(out_features) * 0.01
        self.f = activation()

    # X: (batch_size, in)
    # W: (out, in)
    # b: (out)
    # U: (batch_size, out)
    def forward(self, X):
        assert X.shape[1] == self.in_features
        U = np.dot(X, self.W.T) + self.b
        return U, self.f.forward(U)

    # Y=f(U), U=W*X+b
    # dL/dU = dL/dY * dY/dU
    # dL/dW = dL/dU * dU/dW
    # dL/db = dL/dU * dU/db
    def backward(self, X, U, dL_dY, lr):
        batch_size = X.shape[0]
        assert X.shape == (batch_size, self.in_features)
        assert U.shape == (batch_size, self.out_features)
        assert dL_dY.shape == (batch_size, self.out_features)

        # dL/dY:  (batch, out)
        # dY/dU:  (out, out)
        # dL/dU:  (batch, out)
        # dL/dU = dL/dY * dY/dU
        dL_dU = self.f.backward(U, dL_dY)

        # dL/dU: (batch, out)
        # dU/dW: (batch, in)
        # dL/dW: (out, in)
        # dL/dW = dL/dU * dU/dW
        #       = dL/dU * X
        dL_dW = np.dot(dL_dU.T, X) / batch_size

        # dL/dU: (batch, out)
        # dU/db: (out)
        # dL/db:  (out)
        # dL/db = dL/dU * dU/db
        #       = dL/dU
        dL_db = np.sum(dL_dU, axis=0) / batch_size

        # dL/dU: (batch, out)
        # dU/dX: (out, in)
        # dL/dX: (batch, in)
        # dL/dX = dL/dU * dU/dX
        #       = dL/dU * W
        dL_dX = np.dot(dL_dU, self.W)

        # 最后更新参数
        self.W -= lr * dL_dW
        self.b -= lr * dL_db

        return dL_dX


class Net:
    def __init__(self, net_info, seed=0):
        np.random.seed(seed)
        self.net = []
        for layer in net_info:
            self.net.append(Mlp(**layer["info"]))

    def _forward(self, X):
        outputs = [{"X": X}]
        for layer in self.net:
            U, X = layer.forward(X)
            outputs[-1]["U"] = U
            outputs.append({"X": X})
        return outputs

    def _backward(self, outputs, dL_dX, lr):
        for i in range(len(self.net) - 1, -1, -1):
            dL_dX = self.net[i].backward(outputs[i]["X"], outputs[i]["U"], dL_dX, lr)

    def train(self, X, Y, lossF, batch_size, lr):
        for i in range(0, len(X), batch_size):
            X_batch = X[i : i + batch_size]
            Y_batch = Y[i : i + batch_size]
            outputs = self._forward(X_batch)

            dL_dX = lossF.backward(Y_batch, outputs[-1]["X"])
            self._backward(outputs, dL_dX, lr)

    def pred(self, X):
        return self._forward(X)[-1]["X"]
