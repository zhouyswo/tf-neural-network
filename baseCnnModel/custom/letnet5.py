import tensorflow as tf
import numpy as np
import dataloder as dl


def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def buildModel(x_image,y_lables):
    w_conv1 = weights([5, 5, 3, 6])
    b_conv1 = bias([6])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2nd layer: conv+relu+max_pool
    w_conv2 = weights([5, 5, 6, 16])
    b_conv2 = bias([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])

    # 3rd layer: 3*full connection
    w_fc1 = weights([7 * 7 * 16, 120])
    b_fc1 = bias([120])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    w_fc2 = weights([120, 84])
    b_fc2 = bias([84])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

    w_fc3 = weights([84, 10])
    b_fc3 = bias([10])
    h_fc3 = tf.nn.softmax(tf.matmul(h_fc2, w_fc3) + b_fc3)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_fc3, labels=y_lables))
    optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    prediction_labels = tf.argmax(h_fc3, axis=1, name="output")
    correct_prediction = tf.equal(tf.cast(prediction_labels, tf.float32), y_lables)
    # accuracy = tf.reduce_mean(correct_prediction)
    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
    return dict(
        x=x_image,
        y=y_lables,
        optimize=optimize,
        correct_prediction=correct_prediction,
        correct_times_in_batch=correct_times_in_batch,
        cost=cost
    )


def trainModel(model, args):
    data, lable = dl.read_img(args.data_dir,args.w,args.h,args.c)

    tf_config = tf.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # tf_config.log_device_placement = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.initialize_all_variables())
        # tf.initialize_all_variables().run()
        for epoch in range(args.epoch):
            print("epoch----------------:{}".format(epoch))
            batch_idxs = min(len(data), args.train_size) // args.batch_size
            for idx in range(int(batch_idxs)):
                batch_files = data[idx * args.batch_size:(idx + 1) * args.batch_size]
                batch_lable = lable[idx * args.batch_size:(idx + 1) * args.batch_size]
                print("epoch----------------:{},filelength:{}".format(epoch,len(batch_files)))
                for i in range(len(batch_files)):
                    sess.run([model['optimize']], feed_dict={
                        model['x']: np.reshape(batch_files[i], (1, args.w, args.h, args.c)),
                        model['y']: [[batch_lable[i]]]
                    })

                epoch_delta = 2
                if idx % epoch_delta == 0:
                    total_batches_in_train_set = 0
                    total_correct_times_in_train_set = 0
                    total_cost_in_train_set = 0.
                    for i in range(len(batch_files)):
                        return_correct_times_in_batch = sess.run(model['correct_times_in_batch'], feed_dict={
                            model['x']: np.reshape(batch_files[i], (1, args.w, args.h, args.c)),
                            model['y']: [[batch_lable[i]]]
                        })
                        mean_cost_in_batch = sess.run(model['cost'], feed_dict={
                            model['x']: np.reshape(batch_files[i], (1, args.w, args.h, args.c)),
                            model['y']:[[batch_lable[i]]]
                        })
                        total_batches_in_train_set += 1
                        total_correct_times_in_train_set += return_correct_times_in_batch
                        total_cost_in_train_set += (mean_cost_in_batch * len(batch_files))

                        acy_on_train = total_correct_times_in_train_set / float(total_batches_in_train_set * len(batch_files))
                        print("acy_on_train:{}   loss_on_train:{:6.2f}".format(acy_on_train * 100.0,total_cost_in_train_set))


def main():
    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
    flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
    flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
    flags.DEFINE_integer("batch_size", 2, "The size of batch images [64]")
    flags.DEFINE_integer("h", 216, "The size of image to use (will be center cropped). [108]")
    flags.DEFINE_integer("w", 216,
                         "The size of image to use (will be center cropped). If None, same value as input_height [None]")
    flags.DEFINE_integer("c", 3, "The size of image to use (will be center cropped). [108]")
    flags.DEFINE_integer("n", 3, "The size of image to use (will be center cropped). [108]")
    flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
    flags.DEFINE_string("data_dir", "E:/TestDatas/flower/train/", "Root directory of dataset [data]")
    FLAGS = flags.FLAGS

    x = tf.placeholder('float', shape=[None, FLAGS.w *  FLAGS.h * FLAGS.c])
    x_image = tf.reshape(x, [-1,  FLAGS.w ,FLAGS.h , FLAGS.c])
    y = tf.placeholder('float', shape=[None, 1])
    y_lables = tf.reshape(y, [-1, 1])
    model = buildModel(x_image,y_lables)
    trainModel(model, FLAGS)


main()