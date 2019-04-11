import sys
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.files import load_cifar10_dataset


def tensor_summary(tensor):
    """显示张量的标量和直方图信息"""
    # 获取张量tensor的name;
    tensor_name = tensor.op.name
    # tf.summary.histogram用来显示直方图信息:一般用来显示训练过程中变量的分布情况;
    tf.summary.histogram(tensor_name + "/activations", tensor)
    # tf.summary.scalar用来显示标量信息;
    tf.summary.scalar(tensor_name + "/sparsity", tf.nn.zero_fraction(tensor))

class ResnetConfig(object):
    """默认超参数配置"""
    # epsilon的值(是BN的参数,BN(Batch Normalization)是一种对输入数据预处理以防止过拟合的方法);
    bn_var_epsilon = 0.001
    # 全连接层的正则项的权重衰减率;
    fc_weight_decay = 0.0002
    # 卷积层的正则项的权重衰减率;
    conv_weight_decay = 0.0002
    # 默认初始化方式(每个卷积或全连接层的参数通过一个零均值与设定方差的正态分布进行初始化);
    initializer = tf.contrib.layers.xavier_initializer

class Resnet(object):
    """深度残差网络"""
    def __init__(self, input_tensor, n, is_training=True, config=ResnetConfig()):
        """-----------变量说明-----------------
        input_tensor: 4-D 输入张量;
        n           : int, 每个residual blocks(残差模块)中block的数量;
        is_training : bool, 如果是True就创建新的变量;否则使用之前的;
        config      : 超参数配置类ResnetConfig的实例;"""
        # 参数初始化赋值;
        self.input = input_tensor
        self.n = n
        self.is_training = is_training
        self.config = ResnetConfig()
        self.__build__model__()

    def __build__model__(self):
        """这个函数相当于主函数:构造ResNet模型;"""
        # 如果正在训练,不应引用之前的参数(参数应当更新);
        if self.is_training:
            # 设定reuse;
            reuse = False
        # 否则;
        else:
            # 用已有的参数;
            reuse = True
            # 记录并保证更新所有层的参数汇总;
        layers = []
        # 第一层;
        with tf.variable_scope("conv0", reuse=reuse):
            # 计算前向传播(卷积层、bn层、relu);
            conv0 = self._conv_bn_relu_layer(self.input, 16, 3, strides=1)
            # 显示张量的标量和直方图信息;
            tensor_summary(conv0)
            # 加入到参数汇总;
            layers.append(conv0)
        # 第一个residual blocks;
        for i in range(self.n):
        # 对每一个block;
            with tf.variable_scope("conv1_%d" % i, reuse=reuse):
                # 如果是第一个block;
                if i == 0:
                    # 前向计算;
                   conv1 = self._residual_block(layers[-1], 16, is_first_block=True)
                    # 不是第一个block;
                else:
                    # 前向计算;
                    conv1 = self._residual_block(layers[-1], 16)
                    #[None, 32, 32, 16] #
                    # 显示张量的标量和直方图信息;
                    tensor_summary(conv1)
                    # 加入到参数汇总;
                    layers.append(conv1)
        # 第二个residual blocks;
        for i in range(self.n):
            # 对每一个block;
            with tf.variable_scope("conv2_%d" % i, reuse=reuse):
                # 前向计算;
                conv2 = self._residual_block(layers[-1], 32)  # [None, 16, 16, 32]
                # 显示张量的标量和直方图信息;
                tensor_summary(conv2)
                # 加入到参数汇总;
                layers.append(conv2)
        # 第三个residual blocks;
        for i in range(self.n):
            # 对每一个block;
            with tf.variable_scope("conv3_%d" % i, reuse=reuse):
                # 前向计算;
                conv3 = self._residual_block(layers[-1], 64)  # [None, 8, 8, 64]
                # 显示张量的标量和直方图信息;
                tensor_summary(conv3)
                # 加入到参数汇总;
                layers.append(conv3)
        # 全连接层;
        with tf.variable_scope("fc", reuse=reuse):
            # 获取输入层(注意是layers的最后一个元素)的维度;
            in_channels = layers[-1].get_shape().as_list()[-1]
            # 计算batch normalization层的前向传播;
            bn = self._batch_normalization_layer(layers[-1], in_channels)
            # 计算relu层的前向传播;
            relu = tf.nn.relu(bn)
            # 计算池化层的前向传播;
            global_pool = tf.reduce_mean(relu, axis=[1, 2])
            # 计算全连接层的前向传播;
            output = self._fc_layer(global_pool, 10)
            # 加入到参数汇总;
            layers.append(output)

        # 设定输出;
        self._output = output
        # 设定预测值(图像的类别);
        self._prediction = tf.cast(tf.argmax(tf.nn.softmax(output), axis=1), tf.int32)

    def _get_variable(self, name, shape, initializer=None, is_fc_layer=False):
        """-----------变量说明-----------------
                目的:创建所有层的参数变量;
                name       : string,变量名;
                shape      : list or tuple,描述变量维度;
                initializer: 默认的初始化方式;
                is_fc_layer: 是否对不同的层使用不同的正则化方式;
                """
        # 如果is_fc_layer为真;
        if is_fc_layer:
            # 采用的正则化变换方式;
            scale = self.config.fc_weight_decay
        else:
            # 采用的正则化变换方式;
            scale = self.config.conv_weight_decay
            # 如果没有特别设定初始化方法;
        if initializer is None:
            # 采用默认的;
            initializer = self.config.initializer()
        # 构造参数变量;
        var = tf.get_variable(name, shape, initializer=initializer,
                              regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
        # 返回结果;
        return var

    def _batch_normalization_layer(self, input_tensor, depth_dim=None):
        """-----------变量说明-----------------
                目的:返回batch normalization层的输出;
                input_tensor: 4-D的输入张量;
                depth_dim   : 输入张量input_tensor的最后一维的维度;
                返回        : 正则化变换后的和输入张量同维度的张量;
                """
        # 如果没有给出depth_dim;
        if depth_dim is None:
            # 手动获取输入张量input_tensor的最后一维的维度;
            depth_dim = input_tensor.get_shape().as_list()[-1]
            # 获取参数:tf.nn.moments返回的mean表示一阶矩,variance则是二阶中心矩;
        mean, variance = tf.nn.moments(input_tensor, axes=[0, 1, 2], keep_dims=False)
        # 获取batch normalization的参数;
        # 获取beta(缩放(scale)系数);
        beta = tf.get_variable("beta", [depth_dim, ], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        # 获取gamma(偏移(offset)系数);
        gamma = tf.get_variable("gamma", [depth_dim, ], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        # 计算输出;
        output_tensor = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma,
                                                  self.config.bn_var_epsilon)
        # 返回结果;
        return output_tensor

    def _fc_layer(self, input_tensor, n_out, n_in=None, activation=tf.identity):
        """-----------变量说明-----------------
                目的:返回全连接层的输出;
                input_tensor: 2-D 输入张量;
                n_in        : int, 输入层神经元数量;
                n_out       : int, 输出层神经元数量;
                activation  : 激活函数(默认使用线性激活函数);
                """
        # 如果未给定输入层神经元数量;
        if n_in is None:
            # 手动获取输入层神经元数量;
            n_in = input_tensor.get_shape().as_list()[-1]

        # 获取参数;
        # 获取权重;
        weights = self._get_variable("fc_weight", [n_in, n_out],
                                     initializer=tf.uniform_unit_scaling_initializer(factor=1.0), is_fc_layer=True)
        # 获取偏置;
        biases = self._get_variable("fc_bias", [n_out, ], initializer=tf.zeros_initializer, is_fc_layer=True)
        # 计算线性输出;
        wx_b = tf.matmul(input_tensor, weights) + biases
        # 计算激活函数输出并返回结果;
        return activation(wx_b)

    def _conv_bn_relu_layer(self, input_tensor, nb_filter, filter_size, strides=1):
        """----------变量说明-----------------
                目的:实现了经卷积层、bn层、relu的输出;
                input_tensor: 4-D的输入张量;
                nb_filter   : int, 卷积核的个数;
                filter_size : int, 卷积核的大小;
                strides     : int, 卷积操作的步长(窗口大小);
                """
        # 输入层的最后一维的维度大小;
        in_channels = input_tensor.get_shape().as_list()[-1]
        # 卷积核参数获取;
        filter = self._get_variable("conv", shape=[filter_size, filter_size, in_channels, nb_filter])
        # 卷积操作的前向传播计算;
        conv = tf.nn.conv2d(input_tensor, filter, strides=[1, strides, strides, 1], padding="SAME")
        # bn层的前向传播计算;
        bn = self._batch_normalization_layer(conv, nb_filter)
        # relu前向传播计算及返回结果;
        return tf.nn.relu(bn)

    def _bn_relu_conv_layer(self, input_tensor, nb_filter, filter_size, strides=1):
        """-----------变量说明-----------------
                目的:实现了经bn层、relu、卷积层的输出(和上一个函数只是次序上不同);
                input_tensor: 4-D的输入张量;
                nb_filter   : int, 卷积核的个数;
                filter_size : int, 卷积核的大小;
                strides     : int, 卷积操作的步长(窗口大小);
                """
        # 输入层的最后一维的维度大小;
        in_channels = input_tensor.get_shape().as_list()[-1]
        # bn层的前向传播计算;
        bn = self._batch_normalization_layer(input_tensor, in_channels)
        # relu前向传播计算;
        relu = tf.nn.relu(bn)
        # 卷积核参数获取;
        filter = self._get_variable("conv", shape=[filter_size, filter_size, in_channels, nb_filter])
        # 卷积操作的前向传播计算;
        conv = tf.nn.conv2d(relu, filter, strides=[1, strides, strides, 1], padding="SAME")
        # 返回结果;
        return conv

    def _residual_block(self, input_tensor, out_channels, is_first_block=False):
        """-----------变量说明-----------------
            目的:实现resnet的residual block的前向传播计算;
            input_tensor  : 4-D的输入张量;
            out_channels  : int, 输出层的维度;
            is_first_block: bool, 标识是否是第一个residual block;
            """
        # 输入层的最后一维的维度大小;
        in_channels = input_tensor.get_shape().as_list()[-1]
        # 补全操作:假如feature map的大小递减,应当设定strides =2;
        #  并且须对输入做平均池化后将结果补全在缺失的维度上;
        #  假如需要补全;
        if in_channels*2 == out_channels:
            # 设定窗口大小;
            strides = 2
            # 输入输出维度不变;
        elif in_channels == out_channels:
            # 设定窗口大小;
            strides = 1
            # 异常处理;
        else:
            # 不应该出现的情况;
            raise ValueError("There is mismatch betwwen input and output channels")
            # 第一个模块的第一层卷积层只实现卷积操作;
            #  在变量域名"block_conv1"中;
        with tf.variable_scope("block_conv1"):
            # 如果是第一个模块;
            if is_first_block:
                # 获取卷积核参数;
                filter = self._get_variable("conv", shape=[3, 3, in_channels, out_channels])
                # 1*1的卷积核的卷积层的前向传播计算;
                conv1 = tf.nn.conv2d(input_tensor, filter, strides=[1, 1, 1, 1], padding="SAME")
                # 不是第一个模块;
            else:
                # 经bn层、relu、卷积层的前向传播计算;
                conv1 = self._bn_relu_conv_layer(input_tensor, out_channels, 3, strides=strides)
                # 第二个卷积层;
        with tf.variable_scope("block_conv2"):
            # 第二个卷积层输出(bn层、relu、卷积层);
            conv2 = self._bn_relu_conv_layer(conv1, out_channels, 3, strides=1)
            # 输入的变换处理(strides>1说明需要做补全操作);
        if strides > 1:
            # 先做平均池化计算;
            pooled_input = tf.nn.avg_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding="VALID")
            # 补全;
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [in_channels//2, in_channels//2]])
            # 无需补全处理;
        else:
            # 输出即输入;
            padded_input = input_tensor
            # 返回结果(经典的H(x)=F(x)+x);
        return conv2 + padded_input

    # @property装饰器就是负责把一个方法变成属性调用;
    @property
    def prediction(self):
        # 返回前向输出;
        return self._prediction

    def get_cost(self, y):
        """-----------变量说明-----------------
            目的:计算训练的代价;
            y: 目标输出张量即正确的标签(1-D, [None]);
            """
        # 断言:y的维度和input的第一维维度相同(不成立的话程序立马终止);
        assert y.get_shape().as_list()[0] == self.input.get_shape().as_list()[0]
        # 计算cross entropy代价;
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._output,labels= y)
        # 返回结果;
        return tf.reduce_mean(cross_entropy)


if __name__ == "__main__":
    # 从cifar10获取训练和测试数据;
    X_train, y_train, X_test, y_test = load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False,path="E:/TestDatas")
    # 获取训练文件路径;
    train_dir = sys.path[0] + "/train_dir"
    # 输入占位符;
    input_tensor = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    # 输出标签占位符;
    y = tf.placeholder(tf.int32, shape=[None,])
    # 创建Resnet实例(residual blocks(残差模块)中block的数量为2);
    resent = Resnet(input_tensor, 2, is_training=True)
    # 定义训练代价;
    cost = resent.get_cost(y)
    # 定义训练句柄;
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    # 定义精度计算句柄;
    accuracy = tf.reduce_mean(tf.cast(tf.equal(resent.prediction, y), tf.float32))
    # 参数初始化;
    init = tf.global_variables_initializer()
    # 操作类实例(一个Session对象将操作节点op封装在一定的环境内运行);
    sess = tf.Session()
    # 运行init操作;
    sess.run(init)
    # 开始训练模型;
    print("Start training...")
    # 迭代次数;
    n_epochs = 10
    # 多次迭代;
    for epoch in range(n_epochs):
        # 每个batch获取一批训练数据(X_train_a, y_train_a);
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size=128, shuffle=True):
            # 运行train_op操作;
            sess.run(train_op, feed_dict={resent.input: X_train_a, y: y_train_a})
            # 设定n_batchs为零;
            n_batchs = 0
            # 初始化正确率;
            acc = 0
        # 跑测试;每个batch获取一批训练数据(X_test_a, y_test_a);
        for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, 128, shuffle=True):
            # 累计正确率;
            acc += sess.run(accuracy, feed_dict={resent.input: X_test_a, y: y_test_a})
            # n_batchs累计;
            n_batchs += 1
            # 输出训练详情;
            print("Epoch {0}, test accuracy {1}".format(epoch, acc/n_batchs))








