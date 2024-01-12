import time
import numpy as np
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def FashionMNIST_DataLoader(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    train_data = datasets.FashionMNIST(
        root="../data",
        train=True,
        URL_Data=True,
        transform=trans,
    )
    test_data = datasets.FashionMNIST(
        root="../data",
        train=False,
        URL_Data=True,
        transform=trans,
    )

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

    return train_dataloader, test_dataloader


class MyDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.features = X
        self.labels = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feat = self.features[index]
        label = self.labels[index]
        if self.transform:
            feat = self.transform(feat)
        if self.target_transform:
            label = self.target_transform(label)
        return feat, label


def init_weights(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)


class MyNetwork(nn.Module):
    def __init__(self, net, device="cuda"):
        super().__init__()
        self.net = net
        self.net.apply(init_weights)
        self.to(device)

    def forward(self, X):
        y = self.net(X)
        return y


def train(dataloader, model, lr, loss_fn=None, optimizer=None, device="cuda"):
    if loss_fn == None:
        loss_fn = nn.CrossEntropyLoss()
    if optimizer == None:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % (num_batches / 3) == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn=None, device="cuda"):
    if loss_fn == None:
        loss_fn = nn.CrossEntropyLoss()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def pred(X, model, device="cuda"):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        y = model(X)
    return y
