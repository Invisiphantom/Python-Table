import struct
import numpy as np
from tqdm import tqdm
from my.net import *


def image_data(file_name):
    with open(file_name, "rb") as f:
        buffer = f.read(16)
        magic, num, rows, cols = struct.unpack(">iiii", buffer)
        buffer = f.read(rows * cols * num)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num, rows, cols)
    return data


def label_data(file_name):
    with open(file_name, "rb") as f:
        buffer = f.read(8)
        magic, num = struct.unpack(">ii", buffer)
        buffer = f.read(num)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num, 1)
    return data


def one_hot(labels, num_classes):
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels


def train_valid_data(mnist_dir="/opt/data/mnist_testdata/"):
    # 读取文件数据
    images = image_data(mnist_dir + "train-images.idx3-ubyte")
    labels = label_data(mnist_dir + "train-labels.idx1-ubyte")

    # 随机打乱数据
    np.random.seed(0)
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    # 分割训练集与验证集 (4:1)
    split = int(len(images) * 0.8)
    train_images, valid_images = images[:split], images[split:]
    train_labels, valid_labels = labels[:split], labels[split:]

    # 添加图片通道
    height, width = images.shape[1], images.shape[2]
    train_images = train_images.reshape(-1, 1, height, width)
    valid_images = valid_images.reshape(-1, 1, height, width)

    # 独热标签数据
    train_labels = one_hot(train_labels, 10)
    valid_labels = one_hot(valid_labels, 10)
    return (train_images, train_labels, valid_images, valid_labels)


def shuffle_data(images, labels):
    np.random.seed(0)
    indices = np.random.permutation(len(images))
    return images[indices], labels[indices]


# 不同的模型深度
# 不同的激活函数
# 不同的dropout比率
# 不同的学习率
# 不同的batch_size
# 不同的权重初始化


def run(net, loss_fn, initial_lr, batch_size, data):
    train_images, train_labels, valid_images, valid_labels = data
    epoch_count = 0  # 总迭代次数
    best_accuracy = 0  # 最佳准确率
    reach_epoch_count = 0  # 达到97%所需迭代次数

    lr_count = 0  # 连续无增长计数
    lr_patience = 10  # 学习率容忍次数
    lr_factor = 0.5  # 学习率衰减因子

    worse_count = 0  # 连续无增长计数
    worse_tolerance = 20  # 无增长容忍次数

    max_epochs = 800  # 最大迭代次数
    pbar = tqdm(range(max_epochs))
    for i in pbar:
        train_images, train_labels = shuffle_data(train_images, train_labels)
        net.train(train_images, train_labels, loss_fn, batch_size, initial_lr)

        _train_labels = net.pred(train_images)
        _valid_labels = net.pred(valid_images)
        _train_loss = loss_fn.forward(train_labels, _train_labels)
        _valid_loss = loss_fn.forward(valid_labels, _valid_labels)
        _valid_accuracy = np.mean(np.argmax(_valid_labels, axis=1) == np.argmax(valid_labels, axis=1)) * 100

        epoch_count += 1
        if _valid_accuracy >= 97 and reach_epoch_count == 0:
            reach_epoch_count = epoch_count

        if _valid_accuracy > best_accuracy:
            lr_count, worse_count = 0, 0
            best_accuracy = _valid_accuracy
        else:
            lr_count += 1
            worse_count += 1
            if lr_count > lr_patience:
                initial_lr *= lr_factor
                lr_count = 0
            if worse_count > worse_tolerance:
                print(f"模型连续{worse_tolerance}次无提升，提前终止训练")
                print(f"准确率97%所需迭代次数: {reach_epoch_count}")
                print(f"最佳迭代次数: {epoch_count-worse_count}")
                print(f"最佳准确率：{best_accuracy:.2f}%")
                print(f"损失函数={loss_fn()} 学习率={initial_lr} 批大小={batch_size}")
                print(net)
                break

        pbar.set_postfix(train_loss=_train_loss, valid_loss=_valid_loss, accuracy=f"{_valid_accuracy:.2f}%", best_accuracy=f"{best_accuracy:.2f}%", lr=initial_lr)
