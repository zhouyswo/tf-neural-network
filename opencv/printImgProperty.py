import cv2 as cv
import numpy as np

#定义输出图片属性
def get_img_info(img):
    print(type(img))  #显示图片类型numpy类型的数组
    #图像矩阵的shape属性表示图像的大小，shape会返回tuple元组，分表表示行、列、颜色通道
    print(img.shape)
    print(img.size)
    print(img.dtype)
    # pixel_data = np.array(img)
    # print(pixel_data)

img = cv.imread("../resource/images/2786001.jpg")
# cv.namedWindow("show", cv.WINDOW_NORMAL)
cv.imshow("show",img)
get_img_info(img)
cv.waitKey(0)
cv.destroyAllWindows()
