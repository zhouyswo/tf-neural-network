import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 随机生成200个点
# 在-0.5~0.5之间随机取200个点 并增加维度 200行1列
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
# 生成随机噪点
noice = np.random.normal(0,0.02,x_data.shape)
y_dara = np.square(x_data)+noice

# 创建占位符
x = tf.placeholder(tf.float32,[None,1]) # 1列 不限行
y = tf.placeholder(tf.float32,[None,1]) # 1列 不限行

# 定义神经网路中间层
weight_l1 = tf.Variable(tf.random_normal([1,10])) # 随机数变量  1行10列
biases_l1 = tf.Variable(tf.zeros([1,10]))
wx_plus_b_l1 = tf.matmul(x, weight_l1) + biases_l1     # 两个矩阵相乘再加一个
l1 = tf.nn.tanh(wx_plus_b_l1)   # 双曲正切函数

# 输出层

weight_l1 = tf.Variable(tf.random_normal([10,1])) # 随机数变量  1行10列
biases_l2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(l1, weight_l1) + biases_l2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#损失函数
loss = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_dara})

    pridicr_val = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_dara)
    plt.plot(x_data,pridicr_val,"r-",lw=5)
    plt.show()
