import dill
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def unpickle(file):
    with open(file, "rb") as f:
        dict = pickle.load(f, encoding="bytes")
    return dict


def one_hot(labels, num_classes):
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels


def train_valid_data(cifar10_dir="/opt/data/cifar10_testdata/"):
    images, labels = [], []

    # 读取文件数据
    for i in range(1, 6):
        data_batch = unpickle(cifar10_dir + f"data_batch_{i}")
        images.append(data_batch[b"data"])
        labels.append(data_batch[b"labels"])
    images, labels = np.concatenate(images), np.concatenate(labels)

    # 随机打乱数据
    np.random.seed(1)
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    # 添加图片宽高与通道
    images = images.reshape(-1, 3, 32, 32)

    # 独热编码标签
    labels = one_hot(labels, 10)
    print(images.shape, labels.shape)

    # 分割训练集与验证集 (4:1)
    split = int(len(images) * 0.8)
    train_images, valid_images = images[:split], images[split:]
    train_labels, valid_labels = labels[:split], labels[split:]

    # 转换图像数据
    train_images = torch.from_numpy(train_images).float()  # (40000, 3, 32, 32)
    valid_images = torch.from_numpy(valid_images).float()  # (10000, 3, 32, 32)

    # 转换标签数据
    train_labels = torch.from_numpy(train_labels).float()  # (40000, 10)
    valid_labels = torch.from_numpy(valid_labels).float()  # (10000, 10)

    train_dataset = TensorDataset(train_images, train_labels)
    valid_dataset = TensorDataset(valid_images, valid_labels)
    return (train_dataset, valid_dataset)


def run(model, loss_fn, initial_lr, batch_size, dataset):
    train_dataset, valid_dataset = dataset
    train_size, test_size = len(train_dataset), len(valid_dataset)

    epoch_count = 0  # 总训练轮数
    best_accuracy = 0  # 最佳准确率

    lr_patience = 10  # 学习率容忍次数
    lr_factor = 0.5  # 学习率衰减因子

    worse_count = 0  # 连续无增长计数
    worse_tolerance = 20  # 无增长容忍次数

    train_dataloader = DataLoader(train_dataset, batch_size, True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False)

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=lr_patience, factor=lr_factor)

    max_epochs = 200
    worse_count = 0
    pbar = tqdm(range(max_epochs))
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

        epoch_count += 1
        train_loss /= train_size
        test_loss /= test_size
        accuracy = accuracy / test_size * 100
        scheduler.step(test_loss)

        if accuracy > best_accuracy:
            worse_count = 0
            best_accuracy = accuracy
        else:
            worse_count += 1
            if worse_count == worse_tolerance:
                print(f"模型连续{worse_tolerance}次无提升，提前终止训练")
                print(f"最佳迭代次数: {epoch_count-worse_count}")
                print(f"最佳准确率：{best_accuracy:.2f}%")
                print(f"损失函数={loss_fn()} 学习率={initial_lr} 批大小={batch_size}")
                print(f"模型参数={sum(p.numel() for p in model.parameters())}")
                print(model)
                break

        pbar.set_postfix(lr=f"{optimizer.param_groups[0]["lr"]:.0e}", train_loss=f"{train_loss:.2e}", test_loss=f"{test_loss:.2e}", accuracy=f"{accuracy:.2f}%", best_accuracy=f"{best_accuracy:.2f}%")
