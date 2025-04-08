import dill
import pickle
import struct
import numpy as np

import torch
from torchvision import datasets

from task2_mnist_torch import MNIST_Net as MNIST_Net_Torch
from task2_cifar import CIFAR_Net as CIFAR_Net_Torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets.MNIST(root="/opt/data", train=True, download=True)
datasets.MNIST(root="/opt/data", train=False, download=True)
datasets.CIFAR10(root="/opt/data", train=True, download=True)
datasets.CIFAR10(root="/opt/data", train=False, download=True)

data_mnist_path = ["/opt/data/MNIST/raw/t10k-images-idx3-ubyte", "/opt/data/MNIST/raw/t10k-labels-idx1-ubyte"]
data_cifar10_path = "/opt/data/cifar-10-batches-py/test_batch"

model_mnist_ethan_path = "task2-mnist-ethan.pkl"
model_mnist_torch_path = "task2-mnist-torch.pth"
model_cifar_path = "task2-cifar.pth"

with open(model_mnist_ethan_path, "rb") as f:
    model_mnist_ethan = dill.load(f)

model_mnist_torch = MNIST_Net_Torch().to(device)
model_mnist_torch.load_state_dict(torch.load(model_mnist_torch_path))

model_cifar = CIFAR_Net_Torch().to(device)
model_cifar.load_state_dict(torch.load(model_cifar_path))

accuracy_mnist_my = model_mnist_ethan.interview(data_mnist_path)
print(f"mnist_my: {accuracy_mnist_my:.2f}%")
print(model_mnist_ethan)
print()

accuracy = model_mnist_torch.interview(data_mnist_path, device)
print(f"mnist_torch: {accuracy:.2f}%")

accuracy = model_cifar.interview(data_cifar10_path, device)
print(f"cifar: {accuracy:.2f}%")
