import inspect  ###获取加载模块路径
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]

"""   
numpy将数组以二进制格式保存到磁盘
np.load和np.save是读写磁盘数组数据的两个主要函数，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为.npy的文件中。
np.save("A.npy",A)   #如果文件路径末尾没有扩展名.npy，该扩展名会被自动加上。
 B=np.load("A.npy")        
item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。
person={'name':'lizhong','age':'26','city':'BeiJing','blog':'www.jb51.net'}    
for x in person.items():
    print x
显示：[（'name'，'lizhong'） etc ]
"""


class Vgg16:
    def __init__(self, vgg16_npy_path=None):  ##构造函数
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)  # 获得Vgg16--该类的路径
            path = os.path.abspath(os.path.join(path, os.pardir))  # os.path.abspath(path) #返回绝对路径，os.pardir为当前目录的父目录
            path = os.path.join(path, "vgg16.npy")  # "vgg16.npy"文件名和路径合并成新路径
            vgg16_npy_path = path
            print(path)  # 打印路径

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        #  print (self.data_dict.type())###查看数据类型
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")  # this->conv1_1------新的变量，this->conv_layer---调用成员函数
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")  ############self.conv1_2相当于一个成员变量
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")  ###预测,prob------全局变量
        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):  ######bottom----rgb,name------"xxx"
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    ######padding='SAME'只是保留边缘信息，还是缩减采样
    def conv_layer(self, bottom, name):  ######bottom----rgb,name------"xxx"
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)  #######得到权重W(四维张量)---vgg16.npy(通过"conv1_1"[0]取出W，也就是filter)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)  #######得到偏差b--vgg16.npy(通过"conv1_1"[1]取出b，也就是bias)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):  #######得到权重W(四维张量)---vgg16.npy(通过"conv1_1"取出W，也就是filter)
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):  #######得到偏差b--vgg16.npy(通过"conv1_1"[1]取出b，也就是bias)
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")