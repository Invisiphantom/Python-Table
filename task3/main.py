import torch
from torchvision import datasets

from task3_mnist_resnet import MNIST_Net
from task3_cifar_resnet import CIFAR_Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets.MNIST(root="/opt/data", train=True, download=True)
datasets.MNIST(root="/opt/data", train=False, download=True)
datasets.CIFAR10(root="/opt/data", train=True, download=True)
datasets.CIFAR10(root="/opt/data", train=False, download=True)

data_mnist_path = ["/opt/data/MNIST/raw/t10k-images-idx3-ubyte", "/opt/data/MNIST/raw/t10k-labels-idx1-ubyte"]
data_cifar10_path = "/opt/data/cifar-10-batches-py/test_batch"

model_mnist_path = "task3-mnist-resnet.pth"
model_cifar_path = "task3-cifar-resnet.pth"

model_mnist = MNIST_Net().to(device)
model_mnist.load_state_dict(torch.load(model_mnist_path))

model_cifar = CIFAR_Net().to(device)
model_cifar.load_state_dict(torch.load(model_cifar_path))

accuracy = model_mnist.interview(data_mnist_path, device)
print(f"mnist_torch: {accuracy:.2f}%")

accuracy = model_cifar.interview(data_cifar10_path, device)
print(f"cifar: {accuracy:.2f}%")
