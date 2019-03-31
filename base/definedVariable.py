import tensorflow as tf

# # 定义变量
# x = tf.Variable([1,2])
#
# # 定义常量
# a = tf.constant([3,4])
#
# # 减法op
# sub = tf.subtract(x, a)
#
# # 加法op
# add = tf.add(x, sub)
#
# # 初始化所有变量
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(sub))
#     print(sess.run(add))

# 变量递增
m1 = tf.Variable(0)  # 定义一个变量 初始化为0
m2 = tf.add(m1, 1)  # 变量加1
m3 = tf.assign(m1, m2)  # 变量赋值
init = tf.global_variables_initializer()  # 初始化所有变量
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(m1))
    for _ in range(6):
        sess.run(m3)  # 调用赋值op
        print(sess.run(m1))