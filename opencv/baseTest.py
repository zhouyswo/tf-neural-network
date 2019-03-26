import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('/opt/workplace/testDatas/data/flower_500/500/sunflowers/45045003_30bbd0a142_m.jpg',0)


#显示一张图片
# cv.imshow('image',img)
# cv.waitKey(0)
# cv.destroyAllWindows()


#Matplotlib显示图像
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
