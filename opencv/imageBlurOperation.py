import cv2 as cv

img = cv.imread("../resource/images/tiger.jpg")
# #均值模糊
# mean = cv.blur(img, (5, 5))
# cv.imshow("mean", mean)
#
# #中值模糊
# midd = cv.medianBlur(img, 5)
# cv.imshow("midd", midd)
#
# #高斯模糊
# gauss = cv.GaussianBlur(img, (5, 5), 2)
# cv.imshow("gauss", gauss)
#
# #双边滤波
# bilateral = cv.bilateralFilter(img, 5, 5, 2)
# cv.imshow("bilateral", bilateral)
#



"""
1.均值模糊函数blur()：定义：blur(src,ksize,dst=None, anchor=None, borderType=None)
定义是有5个参数，但最后三个均为none,所以也就2个参数
  src：要处理的原图像
  ksize: 周围关联的像素的范围：代码中（5，5）就是9*5的大小，就是计算这些范围内的均值来确定中心位置的大小

2.中值模糊函数medianBlur(): 定义：medianBlur(src, ksize, dst=None)
ksize与blur()函数不同，不是矩阵，而是一个数字，例如为5，就表示了5*5的方阵

3.高斯平滑函数GaussianBlur():定义：GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
sigmaX：标准差

4.双边滤波函数bilateralFilter():定义：bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None)
d：邻域直径
sigmaColor：颜色标准差
sigmaSpace：空间标准差
"""

#custorm filter

#自定义模糊函数
import numpy as np
def zi_image(src1):
    kernel1 = np.ones((5, 5), np.float)/25#自定义矩阵，并防止数值溢出
    src2 = cv.filter2D(src1, -1, kernel1)
    cv.imshow("自定义均值模糊", src2)
    kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    src2 = cv.filter2D(src1, -1, kernel2)
    cv.imshow("自定义锐化", src2)
"""
使用的函数为：filter2D()：定义为filter2D(src,depth,kernel)
depth：深度，输入值为-1时，目标图像和原图像深度保持一致
kernel: 卷积核（或者是相关核）,一个单通道浮点型矩阵
  """

zi_image(img)
cv.waitKey(0)
cv.destroyAllWindows()
