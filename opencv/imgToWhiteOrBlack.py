import cv2 as cv
import numpy as np


#图像二值化：基于图像的直方图来实现的，0白色 1黑色

img = cv.imread("../resource/images/cats.jpeg")

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("gray", gray)
#
# #大律法,全局自适应阈值 参数0可改为任意数字但不起作用
# ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# print("阈值：%s" % ret)
# cv.imshow("OTSU", binary)
#
# #TRIANGLE法,，全局自适应阈值, 参数0可改为任意数字但不起作用，适用于单个波峰
# ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
# print("阈值：%s" % ret)
# cv.imshow("TRIANGLE", binary)
#
# #自定义阈值为150,大于150的是白色 小于的是黑色
# ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
# print("阈值：%s" % ret)
# cv.imshow("自定义", binary)
#
#
# #自定义阈值为150,大于150的是黑色 小于的是白色
# ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
# print("阈值：%s" % ret)
# cv.imshow("自定义反色", binary)
#
# #截断 大于150的是改为150  小于150的保留
# ret, binary = cv.threshold(img, 150, 255, cv.THRESH_TRUNC)
# print("阈值：%s" % ret)
# cv.imshow("截断1", binary)
#
# # 截断 小于150的是改为150  大于150的保留
# ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_TOZERO)
# print("阈值：%s" % ret)
# cv.imshow("截断2", binary)

#局部阈值
# def local_image(image):
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     cv.imshow("原来", gray)
#     binary1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
#     cv.imshow("局部1", binary1)
#     binary2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)#高斯处理
#     cv.imshow("局部2", binary2)
# local_image(img)


# 求出图像均值作为阈值来二值化
def custom_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("原来", gray)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w * h])  # 化为一维数组
    mean = m.sum() / (w * h)
    print("mean: ", mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imshow("二值", binary)

custom_image(img)

cv.waitKey(0)
cv.destroyAllWindows()