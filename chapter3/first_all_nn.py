# --*-- coding: utf-8 --*--

import tensorflow as tf
import numpy as np


# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数，这里还是沿用3.4.2小节中给出的神经网络结构
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))


# 在shape的一个维度上使用None可以方便使用不同的batch大小。
# 在训练时把数据分成比较小的bacth，但是在测试时，可以一次性使用全部的数据。
# 当数据集比较小时这样比较方便测试，但是数据集比较大时，将大量数据放入一个batch
# 可能会导致内存溢出。
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')


# 定义神经网络前向传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


# 定义损失函数和反向传播的算法
# 激活函数
y = tf.sigmoid(y)
# 交叉熵
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1))
                + (1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1))) # reduce_mean求平均值
# 优化
backprob = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


# 通过随机函数生成一个模拟数据集
rdm = np.random.RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 定义规则来给出样本的标签。在这里所有的x1+x2<1的样例都被认为是正样本，
# 而其他为负样本。和Tensorflow游乐场中的表示法不太一样的地方是，
# 在这里使用0来表示负样本，1表示正样本。大部分解决分类问题的神经网络都会采用0和1表示法。
Y = [[int(x1+x2<1)] for (x1, x2) in X]


# 创建一个会话来运行Tensorflow程序
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    #初始化变量
    sess.run(init)

    # 打印初始化变量w1, w2
    print(sess.run(w1))
    print(sess.run(w2))

    # 设定训练的轮数
    epoch = 10000
    for i in range(epoch):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(backprob, feed_dict={x: X[start:end], y_:Y[start:end]})

        if i%500 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print('After %d training step(s), cross entropy on all data is %g' % (i, total_cross_entropy))
    
    # 查看训练后的参数w1, w2s
    print(sess.run(w1))
    print(sess.run(w2))

