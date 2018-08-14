import tensorflow as tf


# 使用和保存模型代码一样的方式来声明变量
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')

result = v1 + v2
# result = tf.add(v1, v2)

# init = tf.global_variables_initializer()
# 声明tf.train.Saver()类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    # sess.run(init)

    # 加载已经保存的模型， 并通过已经保存的模型中变量的值来计算加法
    saver.restore(sess, './model/add_model.ckpt')
    print(sess.run(result))


'''
import tensorflow as tf 


# 直接加载持久化的图
saver = tf.train.import_meta_graph('./model/add_model.ckpt')

with tf.Session() as sess:
    # model.ckpt.meta是用来存储graph结构的
    saver.restore(sess, './model/add_model.ckpt.meta)
    
    # 通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))
'''