import tensorflow as tf


v = tf.Variable(0, dtype=tf.float32, name='v')
# 声明变量v， 名称为'v:0'
for variable in tf.global_variables():
    print(variable.name)

ema = tf.train.ExponentialMovingAverage(0.99)
# 计算graph每个变量的指数移动平均值
maintain_average_op = ema.apply(tf.global_variables())  
# 声明指数移动平均值后，tf自动生成一个影子变量表示v的移动平均值
for variable in tf.global_variables():
    print(variable.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    sess.run(tf.assign(v, 10))
    sess.run(maintain_average_op)
    # 保存时tf将v和影子变量都保存下来
    saver.save(sess, './model/ema_model.ckpt')
    print(sess.run([v, ema.average(v)]))
