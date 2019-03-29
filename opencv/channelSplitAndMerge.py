import cv2 as cv
import numpy as np

img = cv.imread("../resource/images/2786001.jpg")
# cv.namedWindow("source", cv.WINDOW_NORMAL)
cv.imshow("source", img)

# #通道分离，输出三个单通道图片
b, g, r = cv.split(img)  # 将彩色图像分割成3个通道
cv.imshow("blue", b)
cv.imshow("green", g)
cv.imshow("red", r)

# 通道合并

# merge = cv.merge([b, g, r])
# cv.imshow("merge", merge)
#
#修改某个通道的值
# img[:, :, 2] = 100
# cv.imshow("单通道", img)

cv.waitKey(0)
cv.destroyAllWindows()
