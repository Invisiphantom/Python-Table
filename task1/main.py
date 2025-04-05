import dill
import struct
import numpy as np

eval_datafile_path = ["mnist_testdata/t10k-images.idx3-ubyte", "mnist_testdata/t10k-labels.idx1-ubyte"]
model_pickle_path = "task1-mnist.pickle"

with open(model_pickle_path, "rb") as f:
    model = dill.load(f)

test_accuracy = model.interview(eval_datafile_path)
print(f"测试准确率: {test_accuracy:.2f}%")