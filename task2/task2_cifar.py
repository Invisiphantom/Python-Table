import os
import sys
import dill
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(".."))
from ethan.utils.cutout import Cutout


# tmux new -s task2_cifar
# python task2_cifar.py 2>&1 | tee task2_cifar.log
# Ctrl+B D  &&  tmux ls  &&  tmux attach -t task2_cifar
# tmux new -s tensorboard && tensorboard --logdir=/opt/logs
writer = SummaryWriter(log_dir="/opt/logs/task2_cifar", flush_secs=30)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# 下载数据集
datasets.CIFAR10(root="/opt/data", train=True, download=True)
cifar10_dir = "/opt/data/cifar-10-batches-py/"


# 用于适配interview接口
class CIFAR_Net(nn.Module):
    class TrainDataset(Dataset):
        def __init__(self, images, labels):
            super().__init__()
            self.images = images
            self.labels = labels
            self.trans = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    Cutout(n_holes=1, length=16),
                ]
            )

        def __getitem__(self, index):
            return self.trans(self.images[index]), self.labels[index]

        def __len__(self):
            return len(self.images)

    class ValidDataset(Dataset):
        def __init__(self, images, labels):
            super().__init__()
            self.images = images
            self.labels = labels
            self.trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        def __getitem__(self, index):
            return self.trans(self.images[index]), self.labels[index]

        def __len__(self):
            return len(self.images)

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

    @staticmethod
    def train_valid_loader(cifar10_dir):
        # 读取文件数据
        images, labels = [], []
        for i in range(1, 6):
            data_batch = CIFAR_Net.unpickle(cifar10_dir + f"data_batch_{i}")
            images.append(data_batch[b"data"])
            labels.append(data_batch[b"labels"])
        images, labels = np.concatenate(images), np.concatenate(labels)

        # 添加图片宽高与通道 独热编码标签
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = CIFAR_Net.one_hot(labels, 10)

        # 分割训练集与验证集 (9:1)
        split = int(len(images) * 0.9)
        train_images, valid_images = images[:split], images[split:]
        train_labels, valid_labels = labels[:split], labels[split:]

        # 构造数据集
        train_dataset = CIFAR_Net.TrainDataset(train_images, train_labels)
        valid_dataset = CIFAR_Net.ValidDataset(valid_images, valid_labels)
        return train_dataset, valid_dataset

    def interview(self, eval_datafile_path, device):
        # 读取文件数据
        data_batch = self.unpickle(eval_datafile_path)
        eval_images, eval_labels = data_batch[b"data"], data_batch[b"labels"]

        # 添加图片宽高与通道 独热编码标签
        eval_images = eval_images.reshape(-1, 3, 32, 32)
        eval_labels = CIFAR_Net.one_hot(eval_labels, 10)

        # 数据预处理
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        eval_images = trans(eval_images)

        self.eval()
        with torch.no_grad():
            eval_images, eval_labels = eval_images.to(device), eval_labels.to(device)
            pred = self.forward(eval_images)
            accuracy = torch.sum(torch.argmax(pred, dim=1) == torch.argmax(eval_labels, dim=1)).item()
        return accuracy / len(eval_labels) * 100

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

        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)
        self.layer4 = self._make_layer(256, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    train_dataset, valid_dataset = CIFAR_Net.train_valid_loader(cifar10_dir)
    train_size, valid_size = len(train_dataset), len(valid_dataset)


    epoch_count = 0  # 总训练轮数
    best_accuracy = 0  # 最佳准确率
    best_valid_loss = np.inf  # 最佳验证损失
    model_path = "task2-cifar.pkl"  # 模型保存路径

    # 如果模型文件存在, 则加载模型参数
    model = CIFAR_Net().to(device)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            pre_model = dill.load(f)
        model.load_state_dict(pre_model.state_dict())
        print("模型文件存在, 加载成功")
    else:
        print("模型文件不存在, 从头训练")


    batch_size = 128  # 批处理大小
    initial_lr = 1e-3  # 初始学习率
    max_epoch = 200  # 最大训练轮数

    train_dataloader = DataLoader(train_dataset, batch_size, True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)


    pbar = tqdm(range(max_epoch))
    for i in pbar:
        train_loss, valid_loss, accuracy = 0, 0, 0

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
                valid_loss += loss_fn(pred, y).item()
                accuracy += torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)).item()

        scheduler.step()
        epoch_count += 1
        train_loss /= train_size
        valid_loss /= valid_size
        accuracy = accuracy / valid_size * 100

        if accuracy > best_accuracy:
            best_accuracy = accuracy

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            with open(model_path, "wb") as f:
                dill.dump(model, f)

        writer.add_scalar("train_loss", train_loss, i)
        writer.add_scalar("valid_loss", valid_loss, i)
        writer.add_scalar("accuracy", accuracy, i)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], i)
        pbar.set_postfix(lr=f'{optimizer.param_groups[0]["lr"]:.1e}', train_loss=f"{train_loss:.2e}", valid_loss=f"{valid_loss:.2e}", accuracy=f"{accuracy:.2f}%", best_accuracy=f"{best_accuracy:.2f}%")
