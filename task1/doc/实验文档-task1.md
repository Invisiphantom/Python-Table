
# 回归: 拟合正弦函数

数据集: [-pi,pi]均匀选取1000个点
模型架构:
```py
Net(
  (0): Mlp(in_features=1, out_features=4)
  (1): Gelu()
  (2): Mlp(in_features=4, out_features=4)
  (3): Gelu()
  (4): Mlp(in_features=4, out_features=1)
)
```

| 超参数   |         |
| -------- | ------- |
| 学习率   | 1       |
| 批次大小 | 10      |
| 迭代次数 | 200     |
| 均分误差 | 1.69e-5 |


### 拟合效果展示:
![](https://img.ethancao.cn/20250405214348968.png)

# 分类: ⼿写数字识别

数据集: 随机分割训练集与验证集 (4:1)
模型架构:
```py
Net(
  (0): Flatten()
  (1): Mlp(in_features=784, out_features=128)
  (2): Relu()
  (3): Dropout(p=0.15)
  (4): Mlp(in_features=128, out_features=10)
  (5): Softmax()
)
```

| 超参数         |     |
| -------------- | --- |
| 初始学习率     | 0.1  |
| 学习率容忍次数 | 20  |
| 学习率衰减因子 | 0.5 |
| 无增长容忍次数 | 50  |
| 批次大小       | 64 |

| 结果             |        |
| ---------------- | ------ |
| 最佳迭代次数     | 180    |
| 最佳验证集准确率 | 98.23% |

### 训练过程误差曲线:
![](https://img.ethancao.cn/20250406172346166.png)

### 识别失败的图片 (预测标签-[准确标签])
![](https://img.ethancao.cn/output.png)

