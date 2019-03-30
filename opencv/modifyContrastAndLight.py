import cv2 as cv
import numpy as np

tiger=cv.imread("../resource/images/tiger.jpg")
cv.imshow("tiger",tiger)
h,w,c=tiger.shape
# newNp=np.zeros([h,w,c],tiger.dtype)
img1 = cv.imread("../resource/images/dog.jpeg")
newtiger = cv.addWeighted(tiger,2,img1,1,-50)
cv.imshow("newtiger",newtiger)
cv.waitKey(0)
cv.destroyAllWindows()
"""
addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype=-1);

一共有七个参数：前4个是两张要合成的图片及它们所占比例，
第5个double gamma起微调作用，
第6个OutputArray dst是合成后的图片，
第七个输出的图片的类型（可选参数，默认-1）
有公式得出两个图片加成输出的图片为：dst=src1*alpha+src2*beta+gamma
"""