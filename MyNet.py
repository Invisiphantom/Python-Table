import pickle
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from MyFunc import *


class Net:
    def __init__(self, net_info, seed=0):
        np.random.seed(seed)
        self.net_info = net_info
        self.net = []
        for layer in net_info:
            self.net.append(layer["module"](**layer.get("param", {})))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.net_info, f)
            for layer in self.net:
                pickle.dump(layer.__class__.__name__, f)
                pickle.dump(layer.__dict__, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.net_info = pickle.load(f)
            self.net = []
            for layer in self.net_info:
                layer_name = pickle.load(f)
                layer_dict = pickle.load(f)
                self.net.append(globals()[layer_name](**layer_dict))

    def _forward(self, X):
        outputs = [X]
        for layer in self.net:
            X = layer.forward(X)
            outputs.append(X)
        return outputs

    def _backward(self, outputs, dL_dY, lr):
        for i in range(len(self.net) - 1, -1, -1):
            dL_dY = self.net[i].backward(outputs[i], dL_dY, lr)

    def train(self, X, Y, lossF, batch_size, lr):
        for i in range(0, len(X), batch_size):
            X_batch = X[i : i + batch_size]
            Y_batch = Y[i : i + batch_size]
            outputs = self._forward(X_batch)

            dL_dY = lossF.backward(Y_batch, outputs[-1])
            self._backward(outputs, dL_dY, lr)

    def pred(self, X):
        for layer in self.net:
            if isinstance(layer, Dropout):
                continue
            X = layer.forward(X)
        return X


class Dropout:
    def __init__(self, p):
        self.p = p

    def forward(self, X):
        self.mask = np.random.rand(*X.shape) > self.p
        return X * self.mask / (1 - self.p)

    def backward(self, X, dL_dY, lr):
        return dL_dY * self.mask / (1 - self.p)


class Mlp:
    def __init__(self, in_features, out_features, W=None, b=None):
        self.in_features = in_features
        self.out_features = out_features

        if W is not None:
            self.W = W
            self.b = b
        else:
            self.W = np.random.randn(out_features, in_features) * 0.01
            self.b = np.random.randn(out_features) * 0.01

    # X: (batch_size, in)
    # W: (out, in)
    # b: (out)
    # Y: (batch_size, out)
    def forward(self, X):
        assert X.shape[1] == self.in_features
        return np.dot(X, self.W.T) + self.b

    # Y=W*X+b
    # dL/dW = dL/dY * dY/dW
    # dL/db = dL/dY * dY/db
    def backward(self, X, dL_dY, lr):
        batch_size = X.shape[0]
        assert X.shape == (batch_size, self.in_features)
        assert dL_dY.shape == (batch_size, self.out_features)

        # dL/dY: (batch, out)
        # dY/dW: (batch, in)
        # dL/dW: (out, in)
        # dL/dW = dL/dY * dY/dW
        #       = dL/dY * X
        dL_dW = np.dot(dL_dY.T, X) / batch_size

        # dL/dY: (batch, out)
        # dY/db: (out)
        # dL/db:  (out)
        # dL/db = dL/dY * dY/db
        #       = dL/dY
        dL_db = np.sum(dL_dY, axis=0) / batch_size

        # dL/dY: (batch, out)
        # dU/dX: (out, in)
        # dL/dX: (batch, in)
        # dL/dX = dL/dY * dU/dX
        #       = dL/dY * W
        dL_dX = np.dot(dL_dY, self.W)

        # 最后更新参数
        self.W -= lr * dL_dW
        self.b -= lr * dL_db

        return dL_dX


class Flatten:
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(self.shape[0], -1)

    def backward(self, X, dL_dY, lr):
        return dL_dY.reshape(self.shape)


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, W=None, b=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if W is not None:
            self.W = W
            self.b = b
        else:
            self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
            self.b = np.random.randn(out_channels) * 0.01

    # Y = conv(X, W) + b
    # X: (batch_size, in_channels, in_height, in_width)
    # W: (out_channels, in_channels, kernel_size, kernel_size)
    # b: (out_channels)
    # Y: (batch_size, out_channels, out_height, out_width)
    def forward(self, X):
        batch_size, in_channels, in_height, in_width = X.shape
        assert in_channels == self.in_channels

        out_height = in_height - self.kernel_size + 1
        out_width = in_width - self.kernel_size + 1

        # 方法1: 卷积的循环实现
        # Y1 = np.zeros((batch_size, self.out_channels, out_height, out_width))
        # for i in range(out_height):
        #     for j in range(out_width):
        #         X_slice = X[:, :, i : i + self.kernel_size, j : j + self.kernel_size]
        #         for k in range(self.out_channels):
        #             Y1[:, k, i, j] = np.sum(X_slice * self.W[k], axis=(1, 2, 3))

        # 方法2: 构建滑动窗口视图
        # X: (batch_size, in_channels, in_height, in_width)
        X = sliding_window_view(X, (self.kernel_size, self.kernel_size), axis=(2, 3)).reshape(
            batch_size,
            in_channels,
            out_height,  # IN - KER +1
            out_width,  # IN - KER +1
            self.kernel_size,  # KER
            self.kernel_size,  # KER
        )

        # X: (batch_size,   in_channels, out_height, out_width, kernel_size, kernel_size)
        # W: (out_channels, in_channels,                        kernel_size, kernel_size)
        # Y: (batch_size, out_height, out_width, out_channels)
        # Y: (batch_size, out_channels, out_height, out_width)
        Y = np.tensordot(X, self.W, axes=([1, 4, 5], [1, 2, 3]))
        Y = Y.transpose(0, 3, 1, 2)

        return Y + self.b[np.newaxis, :, np.newaxis, np.newaxis]

    # Y = conv(X, W) + b
    # dL_dW = conv(X, dL_dY)
    # dL_db = sum(dL_dY)
    # dL_dX = conv(dL_dY^p, W^r)
    def backward(self, X, dL_dY, lr):
        batch_size, in_channels, in_height, in_width = X.shape
        assert in_channels == self.in_channels

        out_height = in_height - self.kernel_size + 1
        out_width = in_width - self.kernel_size + 1

        # 方法1: 卷积的循环实现
        # dL_dW1 = np.zeros(self.W.shape)
        # for i in range(self.kernel_size):
        #     for j in range(self.kernel_size):
        #         X_slice = X[:, :, i : i + out_height, j : j + out_width]
        #         for k in range(self.out_channels):
        #             dL_dW1[k, :, i, j] = np.sum(X_slice * dL_dY[:, k][:, np.newaxis], axis=(0, 2, 3)) / batch_size

        # 方法2: 构建滑动窗口视图
        # dL_dW = conv(X, dL_dY)
        # X:      (batch_size, in_channels, in_height, in_width)
        # dL_dY:  (batch_size, out_channels, out_height, out_width)
        # IN=in_height, KER=out_height, IN-KER+1=kernel_size
        X = sliding_window_view(X, (out_height, out_width), axis=(2, 3)).reshape(
            batch_size,
            in_channels,
            self.kernel_size,  # IN-KER+1
            self.kernel_size,  # IN-KER+1
            out_height,  # KER
            out_width,  # KER
        )

        # X:     (batch_size, in_channels, kernel_size, kernel_size,              out_height, out_width)
        # dL_dY: (batch_size,                                       out_channels, out_height, out_width)
        # dL_dW: (in_channels, kernel_size, kernel_size, out_channels)
        # dL_dW: (out_channels, in_channels, kernel_size, kernel_size)

        dL_dW = np.tensordot(X, dL_dY, axes=([0, 4, 5], [0, 2, 3])) / batch_size
        dL_dW = dL_dW.transpose(3, 0, 1, 2)

        # ----------------------------------------------------------------

        # dL_db = sum(dL_dY)
        # dL_dY: (batch_size, out_channels, out_height, out_width)
        # dL_db: (out_channels)
        dL_db = np.sum(dL_dY, axis=(0, 2, 3)) / batch_size

        # ----------------------------------------------------------------

        # dL_dX = conv(dL_dY^p, W^r)
        # dL_dY:   (batch_size, out_channels, out_height, out_width)
        # dL_dY^p: (batch_size, out_channels, in_height-1+kernel_size, in_width-1+kernel_size)
        # W_r:     (out_channels, in_channels, kernel_size, kernel_size)
        dL_dY = np.pad(dL_dY, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)))
        W_r = np.flip(self.W, axis=(2, 3))

        # 方法1: 卷积的循环实现
        # dL_dX1 = np.zeros((batch_size, in_channels, in_height, in_width))
        # for i in range(in_height):
        #     for j in range(in_width):
        #         dL_dY_slice = dL_dY[:, :, i : i + self.kernel_size, j : j + self.kernel_size]
        #         for k in range(self.in_channels):
        #             dL_dX1[:, k, i, j] = np.sum(dL_dY_slice * W_r[:, k], axis=(1, 2, 3)) / batch_size

        # 方法2: 构建滑动窗口视图
        # dL_dX = conv(dL_dY, W_r)
        # dL_dY: (batch_size, out_channels, in_height-1+kernel_size, in_width-1+kernel_size)
        # W_r:   (out_channels, in_channels, kernel_size, kernel_size)
        # IN=in_height+kernel_size-1, KER=kernel_size, IN-KER+1=in_height
        dL_dY = sliding_window_view(dL_dY, (self.kernel_size, self.kernel_size), axis=(2, 3)).reshape(
            batch_size,
            self.out_channels,
            in_height,  # IN-KER+1
            in_width,  # IN-KER+1
            self.kernel_size,  # KER
            self.kernel_size,  # KER
        )

        # dL_dY: (batch_size, out_channels, in_height, in_width,             kernel_size, kernel_size)
        # W_r:   (            out_channels,                     in_channels, kernel_size, kernel_size)
        # dL_dX: (batch_size, in_height, in_width, in_channels)
        # dL_dX: (batch_size, in_channels, in_height, in_width)
        dL_dX = np.tensordot(dL_dY, W_r, axes=([1, 4, 5], [0, 2, 3])) / batch_size
        dL_dX = dL_dX.transpose(0, 3, 1, 2)

        # 最后更新参数
        self.W -= lr * dL_dW
        self.b -= lr * dL_db

        return dL_dX


class MaxPool2d:
    def __init__(self, pool_height, pool_width):
        self.ph, self.pw = pool_height, pool_width

    def forward(self, X):
        batch_size, channels, in_height, in_width = X.shape

        out_height = in_height // self.ph
        out_width = in_width // self.pw

        Y = np.zeros((batch_size, channels, out_height, out_width))
        self.max_index = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)

        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end = i * self.ph, i * self.ph + self.ph
                w_start, w_end = j * self.pw, j * self.pw + self.pw
                X_slice = X[:, :, h_start:h_end, w_start:w_end]

                Y[:, :, i, j] = np.max(X_slice, axis=(2, 3))

                max_indices = np.argmax(X_slice.reshape(batch_size, channels, -1), axis=2)
                h_idx, w_idx = np.unravel_index(max_indices, (self.ph, self.pw))
                self.max_index[:, :, i, j, 0] = h_idx
                self.max_index[:, :, i, j, 1] = w_idx

        return Y

    def backward(self, X, dL_dY, lr):
        batch_size, channels, out_height, out_width = dL_dY.shape

        dL_dX = np.zeros(X.shape)

        for i in range(out_height):
            for j in range(out_width):
                h_start, w_start = i * self.ph, j * self.pw
                h_idx, w_idx = self.max_index[:, :, i, j, 0], self.max_index[:, :, i, j, 1]
                for b in range(batch_size):
                    for c in range(channels):
                        dL_dX[b, c, h_start + h_idx[b, c], w_start + w_idx[b, c]] = dL_dY[b, c, i, j]

        return dL_dX
