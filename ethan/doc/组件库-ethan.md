

# 组件库-ethan

使用numpy实现的神经网络组件库
- ethan.fn: 包含常用的激活函数和损失函数
  - 激活函数: Relu, Gelu, Sigmoid, Softmax
  - 损失函数: MSE, CrossEntropy
- ethan.numpy.net: 包含常用的网络结构
  - 线性层-MLP
  - 消融层-Dropout
  - 展开层-Flatten
  - 二维卷积层-Conv2d
  - 最大池化层-MaxPool2d


# 激活函数

## Relu函数

$\text{Relu}(x)=\max(0,x)$  
$\text{Relu}'(x)=\begin{cases}1 & x>0\\0 & x\leq 0\end{cases}$

**优势**
- **计算简单**: 只需比较大小, 计算量小
- **稀疏性**: 输出中有大量的零值, 增加了网络的稀疏性
- **梯度不消失**: 在正区间, 梯度恒为1, 避免了梯度消失问题
- **收敛速度快**: 在深度网络中表现优异, 训练速度较快

**缺陷**
- **死神经元问题**: 如果输入值始终为负, 神经元将无法更新
- **输出非零均值**: 输出值非负, 可能导致偏置项的累积, 影响训练效果


## Gelu函数

$K=0.044715$  
$\text{Gelu}(x)=\frac{1}{2}x(1+\tanh(\sqrt{2/\pi}(x+Kx^3)))$  

$\text{Gelu}'(x)=\frac{1}{2}(1+\tanh(\sqrt{2/\pi}(x+Kx^3)))$  
$+\frac{1}{2}x(1-\tanh^2(\sqrt{2/\pi}(x+Kx^3)))\cdot\sqrt{2/\pi}(1+0.134145x^2)$

**优势**
- **平滑性**: 相较于Relu, Gelu是连续且平滑的, 能更好地捕捉复杂的非线性关系
- **性能提升**: 在某些任务中, Gelu 比 Relu 和 Sigmoid 表现更优
- **梯度稳定**: 梯度变化平滑, 训练更稳定

**缺陷**
- **计算复杂**: 需要计算 `tanh` 和多项式, 计算量较大
- **硬件支持**: 相较于Relu, 硬件加速支持较少


## Sigmoid函数

$\text{Sigmoid}(x)=\frac{1}{1+e^{-x}}$  
$\text{Sigmoid}'(x)=\frac{e^{-x}}{(1+e^{-x})^2}=\sigma(x)(1-\sigma(x))$

**优势**
- **输出范围明确**: 将输入值映射到 (0, 1), 适合处理概率问题
- **平滑性**: 输出连续且平滑, 适合浅层网络

**缺陷**
- **梯度消失**: 在输入值较大或较小时, 梯度趋近于零, 影响深层网络训练
- **计算复杂**: 需要计算指数函数, 计算量较大
- **非零均值**: 输出值非零均值, 可能导致梯度更新效率降低


## Softmax函数

$\text{Softmax}(x_i)=\frac{e^{x_i}}{\sum e^{x_t}}$  

$\frac{\partial y_i}{\partial x_j}=\begin{cases}y_i(1-y_j) & i=j\\-y_i\cdot y_j & i\neq j\end{cases}$

**优势**
- **概率分布**: 将输出值归一化为概率分布, 适合分类任务
- **多类别支持**: 能处理多类别分类问题
- **梯度明确**: 梯度计算清晰, 适合反向传播

**缺陷**
- **数值稳定性**: 指数运算可能导致数值溢出, 需要额外处理（如减去最大值）
- **梯度消失**: 在类别数较多时, 梯度可能变得很小
- **计算复杂**: 需要计算指数和归一化, 计算量较大


# 损失函数

## 均方误差-MSE

$\text{MSE}(Y,\hat{Y})=\frac{1}{N}\sum(Y-\hat{Y})^2$  
$\text{MSE}'(Y,\hat{Y})=2(Y-\hat{Y})$

**优势**
- **简单直观**: 适合回归任务, 易于理解和实现
- **平滑性**: 损失函数连续且可导, 适合梯度下降优化

**缺陷**
- **对异常值敏感**: 异常值会显著影响损失值, 导致模型偏差
- **不适合分类任务**: 对于分类问题, MSE 的梯度可能不够明确


## 交叉熵-CrossEntropy

$\text{CrossEntropy}(Y,\hat{Y})=-\frac{1}{N}\sum Y\log\hat{Y}$  
$\text{CrossEntropy}'(Y,\hat{Y})=-\frac{Y}{\hat{Y}}$

**优势**
- **适合分类任务**: 能有效衡量预测概率与真实分布之间的差异
- **梯度明确**: 对于分类问题, 梯度计算清晰, 优化效果好
- **数值稳定性**: 常与 Softmax 结合, 避免梯度消失问题

**缺陷**
- **对错误预测敏感**: 如果预测概率接近零, 损失值会非常大
- **需要数值稳定处理**: 需要对预测概率进行裁剪（如加 `epsilon`）以避免数值问题


# 线性层-MLP

## 反向传播-数学公式推导

第一层: $X_1=\sigma(U_0),\ U_0=W_0X_0+b_0$
第二层: $X_2=\sigma(U_1),\ U_1=W_1X_1+b_1$
损失函数: $L=\text{lossF}(X_2,Y)$

首先对第二层求偏导:
$\begin{cases}
    \frac{\partial L}{\partial W_1}=\frac{\partial L}{\partial X_2}\cdot\frac{\partial X_2}{\partial U_1}\frac{\partial U_1}{\partial W_1}&=l'(X_2,Y)\cdot\sigma'(U_1)\cdot X_1 \\
    \frac{\partial L}{\partial b_1}=\frac{\partial L}{\partial X_2}\cdot\frac{\partial X_2}{\partial U_1}\frac{\partial U_1}{\partial b_1}&=l'(X_2,Y)\cdot\sigma'(U_1) \\
    \frac{\partial L}{\partial X_1}=\frac{\partial L}{\partial X_2}\cdot\frac{\partial X_2}{\partial U_1}\frac{\partial U_1}{\partial X_1}&=l'(X_2,Y)\cdot\sigma'(U_1)\cdot W_1 \\
\end{cases}$

然后对第一层求偏导:
$\begin{cases}
    \frac{\partial L}{\partial W_0}=\frac{\partial L}{\partial X_1}\cdot\frac{\partial X_1}{\partial U_0}\frac{\partial U_0}{\partial W_0}&=\frac{\partial L}{\partial X_1}\cdot\sigma'(U_0)\cdot X_0 \\
    \frac{\partial L}{\partial b_0}=\frac{\partial L}{\partial X_1}\cdot\frac{\partial X_1}{\partial U_0}\frac{\partial U_0}{\partial b_0}&=\frac{\partial L}{\partial X_1}\cdot\sigma'(U_0) \\
\end{cases}$

## 反向传播-算法代码实现

```py
class Mlp:
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
```

# 消融层-Dropout

```py
class Dropout:
    def forward(self, X):
        # 记录位置掩码, 以便在反向传播时使用
        self.mask = np.random.rand(*X.shape) > self.p
        return X * self.mask / (1 - self.p)

    def backward(self, X, dL_dY, lr):
        return dL_dY * self.mask / (1 - self.p)
```

# 二维卷积层-Conv2d

## 反向传播-数学公式推导

> 参考链接: [Convolutions and Backpropagations](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)

![](https://img.ethancao.cn/20250314165057.png)

正向传播-滑动窗口加权求和计算:
- $O_{11}=X_{11}F_{11}+X_{12}F_{12}+X_{21}F_{21}+X_{22}F_{22}$
- $O_{12}=X_{12}F_{11}+X_{13}F_{12}+X_{22}F_{21}+X_{23}F_{22}$
- $O_{21}=X_{21}F_{11}+X_{22}F_{12}+X_{31}F_{21}+X_{32}F_{22}$
- $O_{22}=X_{22}F_{11}+X_{23}F_{12}+X_{32}F_{21}+X_{33}F_{22}$

------

![](https://img.ethancao.cn/20250314165258.png)

反向传播-对权重求偏导: 输入值与输出偏导的卷积
公式: $\mathbf{\frac{\partial L}{\partial F}} =\mathbf{\text{conv}(X,\ \frac{\partial L}{\partial O})}$

推导示例: $\begin{aligned}
\frac{\partial L}{\partial F_{11}}&=\frac{\partial L}{\partial O_{11}}\cdot\frac{\partial O_{11}}{\partial F_{11}}+\frac{\partial L}{\partial O_{12}}\cdot\frac{\partial O_{12}}{\partial F_{11}}+\frac{\partial L}{\partial O_{21}}\cdot\frac{\partial O_{21}}{\partial F_{11}}+\frac{\partial L}{\partial O_{22}}\cdot\frac{\partial O_{22}}{\partial F_{11}} \\
    &=\frac{\partial L}{\partial O_{11}}\cdot X_{11}+\frac{\partial L}{\partial O_{12}}\cdot X_{12}+\frac{\partial L}{\partial O_{21}}\cdot X_{21}+\frac{\partial L}{\partial O_{22}}\cdot X_{22} \\
\end{aligned}$

------

![](https://img.ethancao.cn/20250314165057.png)
![](https://img.ethancao.cn/20250404.png)

反向传播-对输入求偏导: 输出偏导填充与权重旋转的卷积
公式: $\mathbf{\frac{\partial L}{\partial X}} =\mathbf{\text{conv}(\frac{\partial L}{\partial O}^{pad},\ F^{flip})}$

| 依赖关系                  | 对输入求偏导                                                                                                                               |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| $X_{11}\to O_{11}$        | $\frac{\partial L}{\partial X_{11}}=\frac{\partial L}{\partial O_{11}}\cdot F_{11}$                                                |
| $X_{12}\to O_{11},O_{12}$ | $\frac{\partial L}{\partial X_{12}}=\frac{\partial L}{\partial O_{11}}\cdot F_{12}+\frac{\partial L}{\partial O_{12}}\cdot F_{11}$ |
| $X_{13}\to O_{12}$        | $\frac{\partial L}{\partial X_{13}}=\frac{\partial L}{\partial O_{12}}\cdot F_{12}$                                                |
| $X_{21}\to O_{11},O_{21}$ | $\frac{\partial L}{\partial X_{21}}=\frac{\partial L}{\partial O_{11}}\cdot F_{21}+\frac{\partial L}{\partial O_{21}}\cdot F_{11}$ |
| $X_{22}\to O_{11},O_{12},O_{21},O_{22}$ | $\frac{\partial L}{\partial X_{22}}=\frac{\partial L}{\partial O_{11}}\cdot F_{22}+\frac{\partial L}{\partial O_{12}}\cdot F_{21}+\frac{\partial L}{\partial O_{21}}\cdot F_{12}+\frac{\partial L}{\partial O_{22}}\cdot F_{11}$ |
| $X_{23}\to O_{12},O_{22}$ | $\frac{\partial L}{\partial X_{23}}=\frac{\partial L}{\partial O_{12}}\cdot F_{22}+\frac{\partial L}{\partial O_{22}}\cdot F_{12}$ |
| $X_{31}\to O_{21}$        | $\frac{\partial L}{\partial X_{31}}=\frac{\partial L}{\partial O_{21}}\cdot F_{21}$                                                |
| $X_{32}\to O_{21},O_{22}$ | $\frac{\partial L}{\partial X_{32}}=\frac{\partial L}{\partial O_{21}}\cdot F_{22}+\frac{\partial L}{\partial O_{22}}\cdot F_{21}$ |
| $X_{33}\to O_{22}$        | $\frac{\partial L}{\partial X_{33}}=\frac{\partial L}{\partial O_{22}}\cdot F_{22}$                                                |


## 反向传播-算法代码实现

```py
class Conv2d:
    def backward(self, X, dL_dY, lr):
        batch_size, in_channels, in_height, in_width = X.shape
        assert in_channels == self.in_channels

        out_height = in_height - self.kernel_size + 1
        out_width = in_width - self.kernel_size + 1

        # dL_dW = conv(X, dL_dY)
        # X:      (batch_size, in_channels, in_height, in_width)
        # dL_dY:  (batch_size, out_channels, out_height, out_width)
        dL_dW = np.zeros(self.W.shape)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                X_slice = X[:, :, i : i + out_height, j : j + out_width]
                for k in range(self.out_channels):
                    # X_slice:                     (batch_size, in_channels, out_height, out_width)
                    # dL_dY[:, k][:, np.newaxis] : (batch_size, 1, out_height, out_width)
                    dL_dW[k, :, i, j] = np.sum(X_slice * dL_dY[:, k][:, np.newaxis], axis=(0, 2, 3)) / batch_size

        # ----------------------------------------------------------------

        # dL_db = sum(dL_dY)
        # dL_dY: (batch_size, out_channels, out_height, out_width)
        dL_db = np.sum(dL_dY, axis=(0, 2, 3)) / batch_size

        # ----------------------------------------------------------------

        # dL_dX = conv(dL_dY^p, W^r)
        # dL_dY:   (batch_size, out_channels, out_height, out_width)
        # dL_dY^p: (batch_size, out_channels, out_height+2*(kernel_size-1), out_width+2*(kernel_size-1))
        #          (batch_size, out_channels, in_height + kernel_size - 1, out_width + kernel_size - 1)
        # W_r:     (out_channels, in_channels, kernel_size, kernel_size)
        # dL_dX:   (batch_size, in_channels, in_height, in_width)
        dL_dY = np.pad(dL_dY, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)))
        W_r = np.flip(self.W, axis=(2, 3))

        dL_dX = np.zeros((batch_size, in_channels, in_height, in_width))
        for i in range(in_height):
            for j in range(in_width):
                dL_dY_slice = dL_dY[:, :, i : i + self.kernel_size, j : j + self.kernel_size]
                for k in range(self.in_channels):
                    # dL_dY_slice: (batch_size, out_channels, kernel_size, kernel_size)
                    # W_r[:, k]:   (out_channels, kernel_size, kernel_size)
                    dL_dX[:, k, i, j] = np.sum(dL_dY_slice * W_r[:, k], axis=(1, 2, 3)) / batch_size

        # 最后更新参数
        self.W -= lr * dL_dW
        self.b -= lr * dL_db

        return dL_dX
```


# 最大池化层-MaxPool2d

## 算法原理

- 正向传播: 对每个窗口取最大值, 并记录其位置
- 反向传播: 仅对最大值位置的梯度进行传递, 其他位置的梯度为0

```py
class MaxPool2d:
    def forward(self, X):
        ......
        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end = i * self.ph, i * self.ph + self.ph
                w_start, w_end = j * self.pw, j * self.pw + self.pw
                X_slice = X[:, :, h_start:h_end, w_start:w_end]

                # 对窗口取最大值
                Y[:, :, i, j] = np.max(X_slice, axis=(2, 3))

                # 记录窗口最大值位置
                max_indices = np.argmax(X_slice.reshape(batch_size, channels, -1), axis=2)
                h_idx, w_idx = np.unravel_index(max_indices, (self.ph, self.pw))
                self.max_index[:, :, i, j, 0] = h_idx
                self.max_index[:, :, i, j, 1] = w_idx
        return Y

    def backward(self, X, dL_dY, lr):
        ......
        for i in range(out_height):
            for j in range(out_width):
                h_start, w_start = i * self.ph, j * self.pw
                h_idx, w_idx = self.max_index[:, :, i, j, 0], self.max_index[:, :, i, j, 1]
                for b in range(batch_size):
                    for c in range(channels):
                        # 只对先前最大值位置的梯度进行传递
                        dL_dX[b, c, h_start + h_idx[b, c], w_start + w_idx[b, c]] = dL_dY[b, c, i, j]
        return dL_dX
```

