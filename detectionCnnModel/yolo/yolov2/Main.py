# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/5/16$ 17:17$
# @Author  : KOD Chen
# @Email   : 821237536@qq.com
# @File    : Main$.py
# Description :YOLO_v2主函数.
# --------------------------------------

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from detectionCnnModel.yolo.yolov2.model_darknet19 import darknet
from detectionCnnModel.yolo.yolov2.decode import decode
from detectionCnnModel.yolo.yolov2.utils import preprocess_image, postprocess, draw_detection
from detectionCnnModel.yolo.yolov2.config import anchors, class_names

def main():
    input_size = (416,416)
    image_file = 'car.jpg'
    image = cv2.imread(image_file)
    image_shape = image.shape[:2] #只取wh，channel=3不取

    # copy、resize416*416、归一化、在第0维增加存放batchsize维度
    image_cp = preprocess_image(image,input_size)

    # 【1】输入图片进入darknet19网络得到特征图，并进行解码得到：xmin xmax表示的边界框、置信度、类别概率
    tf_image = tf.placeholder(tf.float32,[1,input_size[0],input_size[1],3])
    model_output = darknet(tf_image) # darknet19网络输出的特征图
    output_sizes = input_size[0]//32, input_size[1]//32 # 特征图尺寸是图片下采样32倍
    output_decoded = decode(model_output=model_output,output_sizes=output_sizes,
                               num_class=len(class_names),anchors=anchors)  # 解码

    model_path = "./checkpoint_dir"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint(model_path))
        bboxes,obj_probs,class_probs = sess.run(output_decoded,feed_dict={tf_image:image_cp})

    # 【2】筛选解码后的回归边界框——NMS(post process后期处理)
    bboxes,scores,class_max_index = postprocess(bboxes,obj_probs,class_probs,image_shape=image_shape)

    # 【3】绘制筛选后的边界框
    img_detection = draw_detection(image, bboxes, scores, class_max_index, class_names)
    cv2.imwrite("/detection.jpg", img_detection)
    print('YOLO_v2 detection has done!')
    cv2.imshow("detection_results", img_detection)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
