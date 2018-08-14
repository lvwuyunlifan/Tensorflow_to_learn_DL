'''
可用于迁移学习
'''

import tensorflow as tf
from tensorflow.python.platform import gfile


with tf.Session() as sess:
    model_filename = './model/variable_graph_pb.pb'
    # 读取保存的模型文件， 并将文件解析成对应的GraphDef Protool Buffer。
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将graph_def 中保存的图加载到当前的图中。
    # return_elements=['add:0']给出了返回的张量的名称。保存是节点名称为'add'。
    # 在加载的时候输出的是张量的名称，所以是add:0。
    result = tf.import_graph_def(graph_def, return_elements=['add:0'])
    print(sess.run(result))