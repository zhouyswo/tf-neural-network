import cv2 as cv

#高斯金字塔
def pyramid_image(image):
    level = 3  #金字塔的层数
    temp = image.copy()
    pyramid_image = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_image.append(dst)
        # cv.imshow("gauss tower"+str(i), dst)
        temp = dst.copy()
    return pyramid_image

#拉普拉斯金字塔
#拉普拉斯金字塔时，图像大小必须是2的n次方*2的n次方，不然会报错
def  laplian_image(image):
    pyramid_images = pyramid_image(image)
    level = len(pyramid_images)
    for i in range(level-1,-1,-1):
        if (i-1)<0:
            expand= cv.pyrUp(pyramid_images[i],dstsize=image.shape[:2])
            lpls = cv.subtract(image,expand)
            cv.imshow("laplian tower" + str(i), lpls)
        else:
            expand =  cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow("laplian tower"+str(i), lpls)


src = cv.imread("../resource/images/tiger.jpg")
cv.imshow("old", src)
laplian_image(src)
cv.waitKey(0)
cv.destroyAllWindows()