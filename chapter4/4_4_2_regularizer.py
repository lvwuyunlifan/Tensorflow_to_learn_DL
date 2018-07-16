#--*--coding: utf-8 --*--

import tensorflow as tf
from numpy.random import RandomState 


# 获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为"losses"的集合中
def get_weight(shape, lambd):
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # add_to_collection函数将这个新生成变量的L2正则化损失加入集合。
    # 这个函数的第一个参数"losses"是这个集合的名称，第二个参数是要加入这个集合的内容。
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
    # 返回生成的变量
    return var

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

batch_size = 8
# 定义了每一层网络中节点的个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时最深层的节点， 开始的时候就是输入层
cur_layer = x
# 当前层节点个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成5层全连接的神经网络结构
for i in range(1, n_layers):
    # layer_dimension[i] 为下一层的节点个数
    next_dimension = layer_dimension[i]
    # 生成当前层中权重的变量， 并将这个变量的L2正则化损失加入计算图的集合中
    weight = get_weight([in_dimension, next_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[next_dimension]))
    # 使用Relu激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层的节点个数更新为当前层节点个数
    in_dimension = layer_dimension[i]

# 在定义神经网络前向传播的同时，已经将这个变量的L2正则化损失加入了图上的集合
# 这里只需要计算刻画模型在训练数据上表现的损失函数。
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

# get_collection返回一个列表，这个列表是所有这个集合中的元素。
# 在这样例中，这些元素就是损失函数的不同部分，将他们加起来就可以得到最终的损失函数
print('collection: ', tf.get_collection('losses'))
loss = tf.add_n(tf.get_collection('losses'))
print('loss； ', loss)

# 添加学习率衰减
# global_step = tf.Variable(0)
# 使用指数衰减的学习率。在minimize函数中传入的global_step将自动更新
# global_step参数，从而使得学习率也得到相应更新
# learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.96, staircase=False)
# 优化器
optimiter = tf.train.AdamOptimizer(0.001).minimize(loss)
# optimiter = tf.train.AdamOptimizer(learning_rate).minimize(loss)

'''
    添加学习率衰减大概12000就可以到达局部最优解，不添加学习率，定义为0.001时，大概20000到达最优解
'''


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

    epoch = 50000
    for i in range(epoch):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        sess.run(optimiter, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x:X, y_:Y}) 
            print('After %d loss is %g' % (i, total_loss))  