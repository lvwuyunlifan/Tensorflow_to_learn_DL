# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# mnist数据集相关的常数
INPUT_NODE = 784        # 输入层的节点数。对于MNIST数据集，这个就等于图片的像素
OUTPUT_NODE = 10        # 输出层的节点数。这个等于类别的数目，所以这里输出层的节点数是10

# 配置神经网络的参数
LAYER1_NODE = 500       # 隐藏层节点数。这里使用只有一个隐藏层的网络结构作为样例
BATCH_SIZE = 100        # 一个训练的batch中的训练数据个数。数字越小时，训练过程越接近
                        # 随机梯度下降；数字越大时，训练越接近梯度下降

LEARNING_RATE = 0.8             # 学习率
LEARNING_RATE_DECAY = 0.99      # 学习率衰减

REGULARIZATION_RATE = 0.0001    # 描述模型复杂度的正则化项在损失函数的系数
TRAINING_STEPS = 10000           # 训练轮数
MOVING_AVERAGE_DECAY = 0.99     # 移动平均指数

# 计算神经网络的前向传播
def forward_prop(input_tensor, average_class, weights1, biases1, weights2, biases2):
    if average_class:
        # 计算移动平均指数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, average_class.average(weights1)) + average_class.average(biases1))
        layer2 = tf.matmul(layer1, average_class.average(weights2)) + average_class.average(biases2)
        return layer2

    else:
        # 计算前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        layer2 = tf.matmul(layer1, weights2) + biases2
        return layer2

# 训练过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算向前传播结果，无移动平均指数
    y = forward_prop(x, None, weights1, biases1, weights2, biases2)
    
    # 定义存储训练的变量，将训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 加入移动平均指数，将训练轮数定义为加快训练早期变量的更新速度
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # tf.trainable_variable返回的集合GraphKeys.TRAINABLE_AVERAGE中元素
    variable_average_op = variable_average.apply(tf.trainable_variables())

    # 得到移动平均指数y_average
    y_average = forward_prop(x, variable_average, weights1, biases1, weights2, biases2)

    # 交叉熵作为损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算当前batch中的所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失。只计算weight
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失+正则化损失
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率, mnist.train.num_examples/BATCH_SIZE需要迭代的次数
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，
    # 又要更新每一个参数的移动平均指数。为了完成多个操作，tensorflow提供了
    # tf.control_dependencies和tf.group
    with tf.control_dependencies([optimizer, variable_average_op]):
        train_op = tf.no_op(name='train')
    # train_op = tf.group(optimizer, variable_average_op)

    # 使用tf.agrmax(y_average, 1)检验移动平均模型前向传播是否正确，
    # y_average是一个batch_size *10 的二维数组，每一行表示一个样例的预测答案，
    # “1”表示选取最大值的操作仅在第一个维度中进行，即每一行选取最大值的对应的下标，
    # 得到一个长度为batch的一维数组。
    # tf.equal()判断两个张量的每一位是否相等，等True，反之。
    correct_prodiction = tf.equal(tf.argmax(y_average, 1), tf.argmax(y_, 1))

    # 这个运算首先将bool型转成实数型，然后计算平均值。这就是，模型在这一组的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prodiction, tf.float32))

    # 初始化回话开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 验证集
        validation_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 测试集
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代
        for i in range(TRAINING_STEPS):
            # 每1000打印验证结果
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validation_feed)
                print('After %d training step, validation accuracy using average model is %g'%(i, validate_acc))

            # 产生这一轮使用的一个batch的训练数据， 并运行训练数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 测试模型
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step, test accuracy using average model is %g'%(TRAINING_STEPS, test_acc))

def main(arg=None):
    mnist = input_data.read_data_sets('./mnist_data', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()