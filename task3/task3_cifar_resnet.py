import os
import sys
 
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
from ethan.net.resnet import ResNet18
from ethan.utils.cutout import Cutout


# tmux new -s cifar
# python task3_cifar_resnet.py 2>&1 | tee task3_cifar_resnet.log
# Ctrl+B D  &&  tmux ls  &&  tmux attach -t cifar
# tmux new -s tensorboard && tensorboard --logdir=/opt/logs
writer = SummaryWriter(log_dir="/opt/logs/task3_cifar_resnet", flush_secs=30)
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

        # 分割训练集与验证集 (4:1)
        split = int(len(images) * 0.8)
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
        eval_images = eval_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        eval_labels = self.one_hot(eval_labels, 10)

        eval_dataset = CIFAR_Net.ValidDataset(eval_images, eval_labels)
        eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

        self.eval()
        with torch.no_grad():
            accuracy = 0
            for X, y in eval_dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.forward(X)
                accuracy += torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)).item()
        return accuracy / len(eval_labels) * 100

    def __init__(self):
        super().__init__()
        self.net = ResNet18(3, 10)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    train_dataset, valid_dataset = CIFAR_Net.train_valid_loader(cifar10_dir)
    train_size, valid_size = len(train_dataset), len(valid_dataset)

    epoch_count = 0  # 总训练轮数
    best_accuracy = 0  # 最佳准确率
    best_valid_loss = np.inf  # 最佳验证损失
    model_path = "task3-cifar-resnet.pth"  # 模型保存路径


    # 如果模型文件存在, 则加载模型参数
    model = CIFAR_Net().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
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
            torch.save(model.state_dict(), model_path)

        writer.add_scalar("train_loss", train_loss, i)
        writer.add_scalar("valid_loss", valid_loss, i)
        writer.add_scalar("accuracy", accuracy, i)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], i)
        pbar.set_postfix(lr=f'{optimizer.param_groups[0]["lr"]:.1e}', train_loss=f"{train_loss:.2e}", valid_loss=f"{valid_loss:.2e}", accuracy=f"{accuracy:.2f}%", best_accuracy=f"{best_accuracy:.2f}%")
