#--*--coding: utf-8 --*--

import tensorflow as tf
from numpy.random import RandomState


bacth_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
# 回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-output')

# 定义了一个单层的神经网络前向传播的过程， 这里就是简单的加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义预测多了和与预测少了的成本
loss_more = 10
loss_less = 1
# 损失函数
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*loss_more, (y_-y)*loss_less))
'''
    tf.greater的输入为两个变量，比较变量的每个元素的大小，返回大True，小False
    tf.where输入三个变量，第一个为选择条件，True时选择第二个参数，False时选第三个参数
'''
# 优化器
optimiter = tf.train.AdamOptimizer(0.001).minimize(loss)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 设置回归的正确值为两个输入的和加上一个随机数。之所以要加上一个随机量是为了加入不可预测的噪音，
# 否则不同的损失函数的意义就不大了，因为不同的损失函数都会在能完全预测正确的时候最低。
# 一般来说噪音为一个均值为0的小量，所以这里的噪音设置为-0.05～0.05的随机数
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    epoch = 10000
    for i in range(epoch):
        start = (i * bacth_size) % dataset_size
        end = min(start+bacth_size, dataset_size)

        sess.run(optimiter, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x:X, y_:Y}) 
            print('After %d loss is %g' % (i, total_loss))  
    print(sess.run(w1))