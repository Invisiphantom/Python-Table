import os
import dill
import pickle
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from task3_resnet import ResidualBlock

# tmux new -s cifar
# python task3_cifar_resnet.py 2>&1 | tee task3_cifar_resnet.log
# Ctrl+B D  &&  tmux ls  &&  tmux attach -t cifar
# tmux new -s tensorboard && tensorboard --logdir=/opt/logs
writer = SummaryWriter(log_dir="/opt/logs/task3_cifar_resnet", flush_secs=30)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"device: {device}")

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

cifar10_dir = "/opt/data/cifar10_testdata/"


# 用于适配interview接口
class CIFAR_Net(nn.Module):
    @staticmethod
    def unpickle(file):
        with open(file, "rb") as f:
            dict = pickle.load(f, encoding="bytes")
        return dict

    @staticmethod
    def one_hot(labels, num_classes):
        one_hot_labels = np.zeros((len(labels), num_classes))
        for i in range(len(labels)):
            one_hot_labels[i, labels[i]] = 1
        return one_hot_labels

    def interview(self, eval_datafile_path, device):
        data_batch = self.unpickle(eval_datafile_path)
        eval_images, eval_labels = data_batch[b"data"], data_batch[b"labels"]
        eval_images = eval_images.reshape(-1, 3, 32, 32)
        eval_labels = CIFAR_Net.one_hot(eval_labels, 10)
        eval_images = torch.from_numpy(eval_images).float()
        eval_labels = torch.from_numpy(eval_labels).float()
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
            # (3,32,32) -> (32,32,32)
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            #
            # (32,32,32) -> (64,16,16) -> (128,8,8) -> (256,4,4)
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            #
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(1024, 10),
        )

        # 权重初始化
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# 读取文件数据
images, labels = [], []
for i in range(1, 6):
    data_batch = CIFAR_Net.unpickle(cifar10_dir + f"data_batch_{i}")
    images.append(data_batch[b"data"])
    labels.append(data_batch[b"labels"])
images, labels = np.concatenate(images), np.concatenate(labels)

# 添加图片宽高与通道 独热编码标签
images = images.reshape(-1, 3, 32, 32)
labels = CIFAR_Net.one_hot(labels, 10)

# 转换为torch张量
images = torch.from_numpy(images).float()
labels = torch.from_numpy(labels).float()

# 分割训练集与验证集 (9:1)
split = int(len(images) * 0.9)
train_images, valid_images = images[:split], images[split:]
train_labels, valid_labels = labels[:split], labels[split:]

# 构造数据集
train_dataset = TensorDataset(train_images, train_labels)
valid_dataset = TensorDataset(valid_images, valid_labels)
train_size, test_size = len(train_dataset), len(valid_dataset)


epoch_count = 0  # 总训练轮数
best_accuracy = 0  # 最佳准确率
model_path = "task3-cifar-resnet.pkl"  # 模型保存路径

model = CIFAR_Net().to(device)
loss_fn = nn.CrossEntropyLoss()

# if os.path.exists(model_path):
#     with open(model_path, "rb") as f:
#         model = dill.load(f)
#     print("模型文件存在, 加载成功")
# else:
#     print("模型文件不存在, 从头训练")
#     model = CIFAR_Net().to(device)
    
    


batch_size = 128  # 批处理大小
initial_lr = 1e-3  # 初始学习率
max_epoch = 800  # 最大训练轮数
warmup_epochs = 5  # 预热轮数

train_dataloader = DataLoader(train_dataset, batch_size, True)
valid_dataloader = DataLoader(valid_dataset, batch_size, False)
optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.0005)

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
    for X, y in train_dataloader:
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
    epoch_count += 1
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
    pbar.set_postfix(lr=f"{optimizer.param_groups[0]["lr"]:.1e}", train_loss=f"{train_loss:.2e}", test_loss=f"{test_loss:.2e}", accuracy=f"{accuracy:.2f}%", best_accuracy=f"{best_accuracy:.2f}%")

print("模型参数:", sum(p.numel() for p in model.parameters()))
print("保存路径:", model_path)
print("当前训练轮数:", epoch_count)
print("最佳准确率:", best_accuracy)
