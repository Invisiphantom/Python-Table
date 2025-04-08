

# MNIST任务-numpy手写实现

数据集: 随机分割训练集与验证集 (4:1)
数据强化: 随机旋转、随机裁剪
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
数据增强: 随机裁剪, 随机翻转, 归一化
```py
def _make_layer(self, in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()

    self.layer1 = self._make_layer(64, 128)
    self.layer2 = self._make_layer(128, 256)
    self.layer3 = self._make_layer(256, 512)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, 10)
```


| 超参数       |                   |
| ------------ | ----------------- |
| 优化器       | AdamW             |
| 学习率调节器 | CosineAnnealingLR |
| 初始学习率   | 1e-3              |
| 批次大小     | 128               |
| 最大迭代数   | 100               |

| 结果         |          |
| ------------ | -------- |
| 训练集误差   | 3.87e-07 |
| 验证集误差   | 4.24e-04 |
| 验证集准确率 | 99.30%   |



# CIFAR-10任务-torch实现

数据集: 随机分割训练集与验证集 (4:1)
数据增强: 随机裁剪, 随机翻转, 随机遮挡, 归一化
```py
def _make_layer(self, in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()

    self.layer1 = self._make_layer(64, 128)
    self.layer2 = self._make_layer(128, 256)
    self.layer3 = self._make_layer(256, 512)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, 10)
```

| 超参数       |                   |
| ------------ | ----------------- |
| 优化器       | AdamW             |
| 学习率调节器 | CosineAnnealingLR |
| 初始学习率   | 1e-3              |
| 批次大小     | 128               |
| 最大迭代数   | 200               |


| 结果         |          |
| ------------ | -------- |
| 训练集误差   | 3.87e-07 |
| 验证集误差   | 4.24e-04 |
| 验证集准确率 | 99.30%   |