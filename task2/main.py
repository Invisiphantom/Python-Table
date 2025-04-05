import dill
import struct
import numpy as np

mnist_data_path = ["mnist_testdata/t10k-images.idx3-ubyte", "mnist_testdata/t10k-labels.idx1-ubyte"]
cifar10_data_path = 'cifar10_testdata/test_batch'

model_pickle_path = "task1-mnist-my.pickle"



# 读取保存的模型
with open(model_pickle_path, "rb") as f:
    model = dill.load(f)

test_accuracy = model.interview(eval_datafile_path)
print(f"测试准确率: {test_accuracy:.2f}%")