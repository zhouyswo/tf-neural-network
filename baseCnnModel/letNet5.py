from __future__ import division,print_function,absolute_import
import numpy as np
import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/opt/workplace/testDatas/data/mnist_png/", one_hot=True)

num_gpu = 1
num_step = 200
learn_rate = 0.001
batch_size = 1024
display_stap = 10

num_input = 784
num_class = 10
dropout=0.75
gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
config = tf.ConfigProto(allow_soft_placement=True,gpu_option=gpu_option)
tf.Session(config=config)

def conv_net(x,n_class,dropout,reuse,is_training):
    with tf.variable_creator_scope('ConvNet',reuse=reuse):
        x = tf.reshape(x,shape=[-1.28,28,1])

    pass