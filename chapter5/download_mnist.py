#coding: utf-8

from tensorflow.examples.tutorials.mnist import input_data


# 载入MNIST数据集，如果指定地址/path/to/MNIST_data下没有已经下载的数据，
# 那么Tensorflow会自动的从网址下载数据
mnist = input_data.read_data_sets('./mnist_data', one_hot=True)

print("Training data size: ", mnist.train.num_examples)
print("validation data size: ", mnist.validation.num_examples)
print("Testing data size: ", mnist.test.num_examples)
print("Example train size: ", mnist.train.images[0])
print("Example train data label: ", mnist.train.labels[0])


# 
batch_size = 100
x_batch, y_batch = mnist.train.next_batch(batch_size)
# 从batch的集合中选取batch_size个训练数据
print("X shape: ", x_batch.shape)
print("Y shape: ", y_batch.shape)

