import tensorflow as tf

m1 = tf.constant([[2,3]])
m2 = tf.constant([[3],[5]])
opr = tf.matmul(m1,m2)# 两个矩阵相乘
with tf.Session() as sess:
    result = sess.run(opr)
    print(result)