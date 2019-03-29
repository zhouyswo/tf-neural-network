import cv2 as cv


# 调用转换函数实现图像色彩空间转换
def color_space_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    #HSV色彩空间说明： H：0-180  S: 0-255 V： 0-255
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)


img = cv.imread("../resource/images/2786001.jpg")
# cv.imshow("source",img)
color_space_demo(img)
cv.waitKey(0)
cv.destroyAllWindows()




