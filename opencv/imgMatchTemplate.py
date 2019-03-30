# -*- coding=GBK -*-
import cv2 as cv
import numpy as np


# ģ��ƥ��:ģ��ƥ����������ڴ�ͼ����Сͼ��Ҳ����˵��һ��ͼ����Ѱ������һ��ģ��ͼ���λ��
"""
matchTemplate(image, templ, method, result=None, mask=None)
image:ԭͼ
templ:ƥ���ģ��
method��ƥ�䷽ʽ
    ƽ����ƥ��CV_TM_SQDIFF�������ߵ�ƽ������ƥ�䣬��õ�ƥ��ֵΪ0
    ��һ��ƽ����ƥ��CV_TM_SQDIFF_NORMED
    ���ƥ��CV_TM_CCORR�������ߵĳ˻�ƥ�䣬��ֵԽ�����ƥ��̶�Խ��
    ��һ�����ƥ��CV_TM_CCORR_NORMED
    ���ϵ��ƥ��CV_TM_CCOEFF�������ߵ����ϵ��ƥ�䣬1��ʾ������ƥ�䣬-1��ʾ����ƥ��
    ��һ�����ϵ��ƥ��CV_TM_CCOEFF_NORMED
    
minMaxLoc()�������Եõ����ƥ��ֵ�����꣬�������Ϊ���Ͻǽǵ㣬
ģ��Ŀ�͸߻����ξ���ƥ���λ��minMaxLoc()�������Եõ����ƥ��ֵ�����꣬
�������Ϊ���Ͻǽǵ㣬ģ��Ŀ�͸߻����ξ���ƥ���λ��

rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
��Ҫȷ���ľ��Ǿ��ε������㣨���Ͻ������½ǣ�����ɫ���ߵ����ͣ������þ�Ĭ�ϣ�
img:ԭͼ
pt1:���Ͻ�
pt2:���½�
color:��ɫ
lineType:�ߵ����ͣ������þ�Ĭ�ϣ�
shift:������С����λ��
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