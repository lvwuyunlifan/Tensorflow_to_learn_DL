'''
tf提供了convert_variables_to_constants函数，通过此函数可以将graph计算图中的
变量及其取值通过常量的方式存储，这样图和值就存放在一个文件中
'''


import tensorflow as tf
from tensorflow.python.framework import graph_util


v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='other-v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='other-v2')
result = v1 + v2

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # 导出当前计算图的GraphDef部分， 只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()

    # 将graph中的变量及其取值转化为常量， 同时将图中不必要的节点去掉。
    # 最后的参数add是节点名称，而add:0是节点的输出
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph_def, ['add'])

    # 将导出模型存入文件
    with tf.gfile.GFile('./model/variable_graph_pb.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
