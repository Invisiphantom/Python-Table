import dill
import struct
import numpy as np

# mnist_path = ["/opt/data/MNIST/raw/train-images-idx3-ubyte", "/opt/data/MNIST/raw/train-labels-idx1-ubyte"]
mnist_path = ["mnist_testdata/t10k-images.idx3-ubyte", "mnist_testdata/t10k-labels.idx1-ubyte"]
model_path = "task1-mnist.pkl"

with open(model_path, "rb") as f:
    model = dill.load(f)

test_accuracy = model.interview(mnist_path)
print(f"测试准确率: {test_accuracy:.2f}%")
print(model)