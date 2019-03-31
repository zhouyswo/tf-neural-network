import tensorflow as tf

# fetch  同时调用多个op
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(6.0)

add = tf.add(input1, input2)
mul = tf.multiply(input1, input3)

with tf.Session() as sess:
    result = sess.run([mul, add])
    print(result)


# Feed 数据以字典形式传入
# 创建占位符

input4 = tf.placeholder(tf.float32)
input5 = tf.placeholder(tf.float32)
output1= tf.multiply(input4, input5)

with tf.Session() as sess:
    print(sess.run(output1, feed_dict={input4: [8], input5: [3]}))