import os
import dill
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from task3_resnet import ResidualBlock

# nohup python task3_mnist_resnet.py > task3_mnist_resnet.log 2>&1 &
# jobs -l    fg %1    Ctrl+Z
# tensorboard --logdir=/opt/logs 
writer = SummaryWriter(log_dir="/opt/logs/task2-mnist-torch", flush_secs=30)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"device: {device}")

mnist_dir = "/opt/data/mnist_testdata/"


# 用于适配interview接口
class MNIST_Net(nn.Module):
    @staticmethod
    def one_hot(labels, num_classes):
        one_hot_labels = np.zeros((len(labels), num_classes))
        for i in range(len(labels)):
            one_hot_labels[i, labels[i]] = 1
        return one_hot_labels

    def interview(self, eval_datafile_path, device):
        eval_images = idx2numpy.convert_from_file(eval_datafile_path[0])
        eval_labels = idx2numpy.convert_from_file(eval_datafile_path[1])
        eval_images = eval_images.reshape(-1, 1, 28, 28)
        eval_labels = self.one_hot(eval_labels, 10)
        eval_images = torch.from_numpy(np.array(eval_images, copy=True)).float()
        eval_labels = torch.from_numpy(np.array(eval_labels, copy=True)).float()
        eval_images, eval_labels = eval_images.to(device), eval_labels.to(device)

        self.eval()
        with torch.no_grad():
            pred = self.forward(eval_images)
            accuracy = torch.sum(torch.argmax(pred, dim=1) == torch.argmax(eval_labels, dim=1)).item()
        return accuracy / len(eval_labels) * 100

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            #
            # (1,28,28) -> (64,14,14) -> (64,7,7)
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            #
            # (64,7,7) -> (64,7,7)
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            #
            # (64,7,7) -> (128,4,4)
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            #
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

        # 权重初始化
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# 读取文件数据
images = idx2numpy.convert_from_file(mnist_dir + "train-images.idx3-ubyte")
labels = idx2numpy.convert_from_file(mnist_dir + "train-labels.idx1-ubyte")

# 添加图片通道 独热标签数据
height, width = images.shape[1], images.shape[2]
images = images.reshape(-1, 1, height, width)
labels = MNIST_Net.one_hot(labels, 10)


# 转换为torch张量
images = torch.from_numpy(images).float()
labels = torch.from_numpy(labels).float()


# 分割训练集与验证集 (4:1)
split = int(len(images) * 0.8)
train_images, valid_images = images[:split], images[split:]
train_labels, valid_labels = labels[:split], labels[split:]

train_dataset = TensorDataset(train_images, train_labels)
valid_dataset = TensorDataset(valid_images, valid_labels)
train_size, test_size = len(train_dataset), len(valid_dataset)


best_accuracy = 0  # 最佳准确率
model_path = "task3-mnist-resnet.pkl"  # 模型保存路径

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = dill.load(f)
    print("模型文件存在, 加载成功")
else:
    print("模型文件不存在, 从头训练")
    model = MNIST_Net().to(device)
loss_fn = nn.CrossEntropyLoss()


batch_size = 128  # 批处理大小
initial_lr = 3e-4  # 初始学习率
max_epoch = 600  # 最大训练轮数
warmup_epochs = 5  # 预热轮数

train_dataloader = DataLoader(train_dataset, batch_size, True)
valid_dataloader = DataLoader(valid_dataset, batch_size, False)
optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.05)

scheduler = lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
        lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch - warmup_epochs),
    ],
    milestones=[warmup_epochs],
)


pbar = tqdm(range(max_epoch))
for i in pbar:
    train_loss, test_loss, accuracy = 0, 0, 0

    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        for X, y in valid_dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            accuracy += torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)).item()

    scheduler.step()
    train_loss /= train_size
    test_loss /= test_size
    accuracy = accuracy / test_size * 100

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        with open(model_path, "wb") as f:
            dill.dump(model, f)

    writer.add_scalar("train_loss", train_loss, i)
    writer.add_scalar("test_loss", test_loss, i)
    writer.add_scalar("accuracy", accuracy, i)
    writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], i)
    pbar.set_postfix(lr=f"{optimizer.param_groups[0]["lr"]:.0e}", train_loss=f"{train_loss:.2e}", test_loss=f"{test_loss:.2e}", accuracy=f"{accuracy:.2f}%", best_accuracy=f"{best_accuracy:.2f}%")

print("模型参数:", sum(p.numel() for p in model.parameters()))
print("保存路径:", model_path)
print("最佳准确率:", best_accuracy)
