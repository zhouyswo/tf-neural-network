import argparse
import os
import numpy as np
import time

import cv2

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from chainercv.datasets import voc_bbox_label_names

import warnings

warnings.filterwarnings('ignore')


def Ex(f):
    print('File Canged' + f)
    # start=time.time()
    root, ext = os.path.splitext(f)

    img = imread(f)
    result = img.copy()
    if img is None:
        raise FileNotFoundError('FileNotFound::' + f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Result image
    # (H, W, C) -> (C, H, W)
    img = np.asarray(img, dtype=np.float32).transpose((2, 0, 1))

    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    if len(bbox) != 0:
        for i, bb in enumerate(bbox):
            # print(i)
            lb = label[i]
            conf = score[i].tolist()
            ymin = int(bb[0])
            xmin = int(bb[1])
            ymax = int(bb[2])
            xmax = int(bb[3])

            x_avg = (xmax + xmin) / 2
            y_avg = (ymax + ymin) / 2

            class_num = int(lb)

            # Draw box 1
            # cv2.rectangle(result, (xmin, ymin), (xmax, ymax),
            # voc_semantic_segmentation_label_colors[class_num], 2)

            # Draw box 2
            cv2.rectangle(result, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            text = voc_bbox_label_names[class_num] + " " + ('%.2f' % conf)

            text_top = (xmin, ymin - 10)
            text_bot = (xmin + 80, ymin + 5)
            text_pos = (xmin + 5, ymin)

            # Draw label 1
            # cv2.rectangle(result, text_top, text_bot,
            # voc_semantic_segmentation_label_colors[class_num], -1)
            # cv2.putText(result, text, text_pos,
            # cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

            # Draw label 2
            cv2.rectangle(result, text_top, text_bot, (255, 255, 255), -1)
            cv2.putText(result, text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    name, _ = os.path.splitext(os.path.basename(f))
    imwrite(path2 + r'/' + name + '.jpg', result)
    # print('Time:'+str(time.time()-start))


def getext(filename):
    return os.path.splitext(filename)[-1].lower()


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


path = '';
path2 = ''
while not os.path.exists(path):
    print("Please type text FilePath")
    path = input(">>")

while not os.path.exists(path2):
    print("Please type Output FilePath \n Default is C:\\DriveRecorder\\test")
    path2 = input(">>")
    if path2 == '':
        path2 = r'C:\DriveRecorder\test'
        break

model = FasterRCNNVGG16(n_fg_class=len(voc_bbox_label_names),
                        pretrained_model="voc07")
chainer.cuda.get_device_from_id(0).use()
print("Runing on GPU")
model.to_gpu()

print('Initialize　Complete')

while True:
    f = open(path)
    before = f.read()
    f.close()
    while True:
        f = open(path)
        after = f.read()
        f.close()
        if before != after:
            Ex(after)
            before = after
time.sleep(1)