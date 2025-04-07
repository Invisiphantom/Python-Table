 
import pickle
import struct
import numpy as np

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# data_mnist_path = ["/opt/data/mnist_testdata/train-images.idx3-ubyte", "/opt/data/mnist_testdata/train-labels.idx1-ubyte"]
# data_cifar10_path = '/opt/data/cifar10_testdata/data_batch_1'

data_mnist_path = ["mnist_testdata/t10k-images.idx3-ubyte", "mnist_testdata/t10k-labels.idx1-ubyte"]
data_cifar10_path = "cifar10_testdata/test_batch"


model_mnist_my_path = "task2-mnist-ethan.pkl"
model_mnist_torch_path = "task2-mnist-torch.pkl"
model_cifar_path = "task2-cifar.pkl"

with open(model_mnist_my_path, "rb") as f:
    model_mnist_my = dill.load(f)

with open(model_mnist_torch_path, "rb") as f:
    model_mnist_torch = dill.load(f).to(device)

with open(model_cifar_path, "rb") as f:
    model_cifar = dill.load(f).to(device)

accuracy_mnist_my = model_mnist_ethan.interview(data_mnist_path)
print(f"mnist_my: {accuracy_mnist_my:.2f}%")
print(model_mnist_my)
print()

accuracy_mnist_torch = model_mnist_torch.interview(data_mnist_path, device)
print(f"mnist_torch: {accuracy_mnist_torch:.2f}%")
print(model_mnist_torch)
print()

accuracy_cifar = model_cifar.interview(data_cifar10_path, device)
print(f"cifar: {accuracy_cifar:.2f}%")
print(model_cifar)
print()
