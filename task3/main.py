import pickle
import struct
import numpy as np

import torch
from torch import nn

from task3_cifar_resnet import CIFAR_Net

import os
import sys

sys.path.append(os.path.abspath(".."))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data_mnist_path = ["mnist_testdata/t10k-images.idx3-ubyte", "mnist_testdata/t10k-labels.idx1-ubyte"]
data_cifar10_path = "/opt/data/cifar-10-batches-py/test_batch"


model_mnist_path = "task3-mnist-resnet.pkl"
model_cifar_path = "task3-cifar-resnet.pth"

# with open(model_mnist_path, "rb") as f:
#     model_mnist_torch = dill.load(f).to(device)

model_cifar = CIFAR_Net().to(device)
model_cifar.load_state_dict(torch.load(model_cifar_path))

# accuracy_mnist_torch = model_mnist_torch.interview(data_mnist_path, device)
# print(f"mnist_torch: {accuracy_mnist_torch:.2f}%")
# print(model_mnist_torch)

accuracy_cifar = model_cifar.interview(data_cifar10_path, device)
print(f"cifar: {accuracy_cifar:.2f}%")
print()
