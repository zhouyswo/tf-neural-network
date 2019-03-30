# -*- coding=GBK -*-
import cv2 as cv
import numpy as np


# 模版匹配:模板匹配就是用来在大图中找小图，也就是说在一副图像中寻找另外一张模板图像的位置
"""
matchTemplate(image, templ, method, result=None, mask=None)
image:原图
templ:匹配的模板
method：匹配方式
    平方差匹配CV_TM_SQDIFF：用两者的平方差来匹配，最好的匹配值为0
    归一化平方差匹配CV_TM_SQDIFF_NORMED
    相关匹配CV_TM_CCORR：用两者的乘积匹配，数值越大表明匹配程度越好
    归一化相关匹配CV_TM_CCORR_NORMED
    相关系数匹配CV_TM_CCOEFF：用两者的相关系数匹配，1表示完美的匹配，-1表示最差的匹配
    归一化相关系数匹配CV_TM_CCOEFF_NORMED
    
minMaxLoc()函数可以得到最大匹配值的坐标，以这个点为左上角角点，
模板的宽和高画矩形就是匹配的位置minMaxLoc()函数可以得到最大匹配值的坐标，
以这个点为左上角角点，模板的宽和高画矩形就是匹配的位置

rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
须要确定的就是矩形的两个点（左上角与右下角）。颜色，线的类型（不设置就默认）
img:原图
pt1:左上角
pt2:右下角
color:颜色
lineType:线的类型（不设置就默认）
shift:坐标点的小数点位数
"""

def template_image():
    tpl = cv.imread("../resource/images/tiger.jpg")
    target = cv.imread("../resource/images/tigerEye.png")
    # cv.imshow("tpl", tpl)
    # cv.imshow("target", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = target.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(tpl,target, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(tpl, tl, br, (0, 0, 255), 2)
        cv.imshow("model" + np.str(md), tpl)


template_image()
cv.waitKey(0)
cv.destroyAllWindows()