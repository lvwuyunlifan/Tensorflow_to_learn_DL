import tensorflow as tf


v = tf.Variable(0, dtype=tf.float32, name='v')
ema = tf.train.ExponentialMovingAverage(0.99)

# 通过使用variables_to_restore函数直接生成上面代码中的提供字典
# {'v/ExponentialMovingAverage':v}
# v就代表了ema
print(ema.variables_to_restore())

saver = tf.train.Saver(ema.variables_to_restore())