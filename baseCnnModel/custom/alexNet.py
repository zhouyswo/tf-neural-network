import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
import dataloder as dl

def convinit(w, h, channel, featurenum):
    W = tf.Variable(tf.truncated_normal([w, h, channel, featurenum], stddev=0.01))  # 首先需要创建W和b变量
    b = tf.Variable(tf.constant(0.01, shape=[featurenum]))
    return W, b


def fcinit(inputD, outputD):
    W = tf.Variable(tf.truncated_normal([inputD, outputD], stddev=0.01), dtype=tf.float32)
    b = tf.Variable(tf.constant(0.01, shape=[outputD]), dtype=tf.float32)
    return W, b


def convLayer(x, W, b, stride_x, stride_y, Flagure, padding='SAME'):
    conv = tf.nn.conv2d(x, W, strides=[1, stride_x, stride_y, 1], padding=padding)  # 进行卷积处理
    out = tf.add(conv, b)
    if Flagure:
        return tf.nn.relu(out)
    else:
        return out  # 在最后一个卷积时不需要用relu


def LRN(x, alpha, beta, R, bias):
    y = tf.nn.local_response_normalization(x, depth_radius=R, alpha=alpha, beta=beta, bias=bias)
    return y


def max_poolLayer(x, w, h, stride_x, stride_y, padding='SAME'):
    y = tf.nn.max_pool(x, ksize=[1, w, h, 1], strides=[1, stride_x, stride_y, 1], padding=padding)
    return y


def dropout(x, keeppro):
    y = tf.nn.dropout(x, keeppro)
    return y


def fcLayer(x, W, b, Flagure):
    out = tf.add(tf.matmul(x, W), b)
    if Flagure:
        return tf.nn.relu(out)
    else:
        return out


def model(x, keeppro):
    # conv1
    W1, b1 = convinit(10, 10, 3, 64)
    conv1 = convLayer(x, W1, b1, 4, 4, True, 'VALID')
    LRN1 = LRN(conv1, 2e-05, 0.75, 2, 1)
    maxpool1 = max_poolLayer(LRN1, 3, 3, 2, 2, 'VALID')
    # conv2
    W2, b2 = convinit(5, 5, 64, 96)
    conv2 = convLayer(maxpool1, W2, b2, 2, 2, True, 'VALID')
    LRN2 = LRN(conv2, 2e-05, 0.75, 2, 1)
    maxpool2 = max_poolLayer(LRN2, 3, 3, 2, 2, 'VALID')
    # conv3
    W3, b3 = convinit(3, 3, 96, 128)
    conv3 = convLayer(maxpool2, W3, b3, 1, 1, True, 'SAME')
    # conv4
    W4, b4 = convinit(3, 3, 128, 256)
    conv4 = convLayer(conv3, W4, b4, 1, 1, True, 'SAME')
    # conv5
    W5, b5 = convinit(3, 3, 256, 256)
    conv5 = convLayer(conv4, W5, b5, 1, 1, True, 'SAME')
    maxpool5 = max_poolLayer(conv5, 2, 2, 2, 2, 'SAME')
    # fclayer1
    fcIn = tf.reshape(maxpool5, [-1, 4 * 4 * 256])
    W_1, b_1 = fcinit(4 * 4 * 256, 512)
    fcout1 = fcLayer(fcIn, W_1, b_1, True)
    dropout1 = dropout(fcout1, keeppro)
    # fclayer2
    W_2, b_2 = fcinit(512, 256)
    fcout2 = fcLayer(dropout1, W_2, b_2, True)
    dropout2 = dropout(fcout2, keeppro)
    # fclayer3
    W_3, b_3 = fcinit(256, 2)
    fcout3 = fcLayer(dropout2, W_3, b_3, False)
    out_1 = tf.nn.softmax(fcout3)
    out = dropout(out_1, keeppro)
    return out


def accuracy(x, y):
    global out
    predict = sess.run(out, feed_dict={x: test_x, keeppro: 0.5})
    correct_predict = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    result = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keeppro: 0.5})
    return predict, result


# make data
# read file
# file = 'D:\\CNN paper\\Alex_net\\image1000test200\\train.txt'
# os.chdir('D:\\CNN paper\\Alex_net\\image1000test200\\train')
# with open(file, 'rb') as f:
#     dirdata = []
#     for line in f.readlines():
#         lines = bytes.decode(line).strip().split('\t')
#         dirdata.append(lines)
# dirdata = np.array(dirdata)
#
# # read imgdata
# imgdir, label_1 = zip(*dirdata)
# alldata_x = []
# for dirname in imgdir:
#     img = cv.imread(dirname.strip(), cv.IMREAD_COLOR)
#     imgdata = cv.resize(img, (320, 320), cv.INTER_LINEAR)
#     alldata_x.append(imgdata)
# # random shuffle
# alldata = zip(alldata_x, label_1)
# temp = list(alldata)
# random.shuffle(temp)
data_xs, data_label = dl.read_img("E:/TestDatas/flower/train/",320, 320, 3)
data_x = np.array(data_xs)
label = [int(i) for i in data_label]
# label one hot
tf_label_onehot = tf.one_hot(label, 2)
with tf.Session() as sess:
    data_y = sess.run(tf_label_onehot)
# data increase
train_x = data_x[:40]
train_y = data_y[:40]
test_x = data_x[40:60]
test_y = data_y[40:60]

x = tf.placeholder(tf.float32, [None, 320, 320, 3])
y = tf.placeholder(tf.float32, [None, 2])
keeppro = tf.placeholder(tf.float32)
out = model(x, keeppro)
out = tf.clip_by_value(out, 1e-10, 1.0)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out), reduction_indices=[1]))
Optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()
if not os.path.exists("alexnet"):
    os.makedirs("alexnet")
with tf.Session() as sess:
    sess.run(init)
    server = tf.train.Saver()
    for i in range(100):
        sess.run(Optimizer, feed_dict={x: train_x, y: train_y, keeppro: 0.5})
        if i % 10 == 0:
            cost = sess.run(loss, feed_dict={x: train_x, y: train_y, keeppro: 0.5})
            server.save(sess,"./alexnet/")
            print('after %d iteration,cost is %f' % (i, cost))
            predict = sess.run(out, feed_dict={x: test_x, keeppro: 0.5})
            correct_predict =np.argmax(predict, axis=1) #tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
            print(correct_predict)