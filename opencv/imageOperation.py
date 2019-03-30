import cv2 as cv

img1 = cv.imread("../resource/images/dog.jpeg")
img2 = cv.imread("../resource/images/tiger.jpg")
#相加,变换前后顺序，生成图像不变
# add = cv.add(img2,img1)
# cv.imshow("add",add)

#相减,变换前后顺序，生成图像不一致
# sub = cv.subtract(img2,img1)
# cv.imshow("sub",sub)

#相乘,变换前后顺序，生成图像不一致
# divide = cv.divide(img2,img1)
# cv.imshow("divide",divide)

#相除,变换前后顺序，生成图像不变
# multip = cv.multiply(img1,img2)
# cv.imshow("multip",multip)

#与,变换前后顺序，生成图像不变
# yu = cv.bitwise_and(img2,img1)
# cv.imshow("yu",yu)

#或,变换前后顺序，生成图像不变
# huo = cv.bitwise_or(img1,img2)
# cv.imshow("huo",huo)

#非,变换前后顺序，生成图像不一致
# not1 = cv.bitwise_not(img2,img1)
# cv.imshow("not",not1)

#异或,变换前后顺序，生成图像不变
xor = cv.bitwise_xor(img1,img2)
cv.imshow("xor",xor)


cv.waitKey(0)
cv.destroyAllWindows()