import cv2 as cv
import numpy as np

#色彩空间转换，利用inrange函数过滤视频中的颜色，实现跟踪某一颜色

def nextrace_object_demo():
    capture = cv.VideoCapture("../resource/video/girl.mp4")
    while True:
        ret,frame = capture.read()
        if ret == False:
            break
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)#转换色彩空间为hsv
        #设置白色的范围，跟踪视频中的白色
        lower_hsv = np.array([0, 0, 51])#设置过滤的颜色的低值
        upper_hsv = np.array([180, 250, 255])#设置过滤的颜色的高值
        #调节图像颜色信息（H）、饱和度（S）、亮度（V）区间，选择白色区域
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        cv.imshow("video", frame)
        cv.imshow("mask", mask)
        if cv.waitKey(50) & 0xFF == ord('q'):
            break

nextrace_object_demo()
cv.waitKey(0)
cv.destroyAllWindows()

'''
HSV色彩空间说明：
H：0-180  S: 0-255 V： 0-255
'''