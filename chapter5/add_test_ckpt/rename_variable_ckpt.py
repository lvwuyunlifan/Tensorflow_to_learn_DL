import tensorflow as tf


# 这里声明的变量和模型变量的名称不同
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='other-v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='other-v2')

# 重命名
saver = tf.train.Saver({'v1':v1, 'v2':v2})

#......