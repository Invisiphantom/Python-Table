

# MNIST任务-numpy手写实现

数据集: 随机分割训练集与验证集 (4:1)
模型架构:
```py
Net(
  (0): Conv2d(in_channels=1, out_channels=2, kernel_size=5)
  (1): Gelu()
  (2): Flatten()
  (3): Mlp(in_features=1152, out_features=256)
  (4): Gelu()
  (5): Dropout(p=0.3)
  (6): Mlp(in_features=256, out_features=10)
  (7): Softmax()
)
```

| 超参数         |     |
| -------------- | --- |
| 初始学习率     | 10  |
| 学习率容忍次数 | 5   |
| 学习率衰减因子 | 0.5 |
| 无增长容忍次数 | 20  |
| 批次大小       | 512 |

| 结果             |       |
| ---------------- | ----- |
| 最佳迭代次数     | 70    |
| 最佳验证集准确率 | 98.7% |

### 训练过程误差曲线:
![](https://img.ethancao.cn/20250405220314228.png)


# MNIST任务-torch实现

数据集: 随机分割训练集与验证集 (4:1)
模型参数量: 421642
```py
(net): Sequential(
  (0): Conv2d(1, 32, kernel_size=3, padding=1)
  (1): ReLU()
  (2): MaxPool2d(kernel_size=2=1)
  (4): ReLU()
  (5): MaxPool2d(kernel_size=2, out_features=128, bias=True)
  (8): ReLU()
  (9): Dropout(p=0.5, inplace=False)
  (10): Linear(in_features=128, out_features=10, bias=True)
)
```

| 超参数         |                   |
| -------------- | ----------------- |
| 优化器         | AdamW             |
| 学习率调节器   | ReduceLROnPlateau |
| 初始学习率     | 1e-3              |
| 学习率容忍次数 | 10                |
| 学习率衰减因子 | 0.5               |
| 无增长容忍次数 | 20                |
| 批次大小       | 128               |

| 结果             |         |
| ---------------- | ------- |
| 最佳迭代次数     | 46      |
| 最佳验证集准确率 | 99.275% |


# CIFAR-10任务-torch实现

数据集: 随机分割训练集与验证集 (4:1)
模型参数量: 5354826
```py
(net): Sequential(
  (0): Conv2d(3, 64, kernel_size=3, padding=1)
  (1): BatchNorm2d(64)
  (2): ReLU()
  (3): Conv2d(64, 64, kernel_size=3, padding=1)
  (4): BatchNorm2d(64)
  (5): ReLU()
  (6): MaxPool2d(kernel_size=2)

  (7): Conv2d(64, 128, kernel_size=3, padding=1)
  (8): BatchNorm2d(128)
  (9): ReLU()
  (10): Conv2d(128, 128, kernel_size=3, padding=1)
  (11): BatchNorm2d(128)
  (12): ReLU()
  (13): MaxPool2d(kernel_size=2)

  (14): Conv2d(128, 256, kernel_size=3, padding=1)
  (15): BatchNorm2d(256)
  (16): ReLU()
  (17): Conv2d(256, 256, kernel_size=3, padding=1)
  (18): BatchNorm2d(256)
  (19): ReLU()
  (20): MaxPool2d(kernel_size=2)

  (21): Flatten()
  (22): Linear(in_features=4096, out_features=1024, bias=True)
  (23): BatchNorm1d(1024)
  (24): ReLU()
  (25): Dropout(p=0.3)
  (26): Linear(in_features=1024, out_features=10, bias=True)
)
```

| 超参数         |                   |
| -------------- | ----------------- |
| 优化器         | AdamW             |
| 学习率调节器   | ReduceLROnPlateau |
| 初始学习率     | 1e-3              |
| 学习率容忍次数 | 10                |
| 学习率衰减因子 | 0.5               |
| 无增长容忍次数 | 20                |
| 批次大小       | 128               |

| 结果             |        |
| ---------------- | ------ |
| 最佳迭代次数     | 65     |
| 最佳验证集准确率 | 85.61% |
