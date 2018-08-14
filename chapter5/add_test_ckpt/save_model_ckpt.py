import tensorflow as tf 


# 声明两个变量并计算他们的和add
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')

result = v1 + v2
# result = tf.add(v1, v2)

init = tf.global_variables_initializer()
# 声明tf.train.Saver()类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # 将模型全部保存到./model/add_model.ckpt
    saver.save(sess, './model/add_model.ckpt')


'''
model.ckpt.meta是用来存储graph结构
model.ckpt是用来存储每一个变量的取值
checkpoint是用来目录下所有的模型文件列表
'''