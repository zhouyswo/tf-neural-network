import cv2 as cv
from matplotlib import pyplot as plt



"""
图像的构成是有像素点构成的，每个像素点的值代表着该点的颜色（灰度图或者彩色图）。
所谓直方图就是对图像的中的这些像素点的值进行统计，得到一个统一的整体的灰度概念。
直方图的好处就在于可以清晰了解图像的整体灰度分布，这对于后面依据直方图处理图像来说至关重要。

一般情况下直方图都是灰度图像，直方图x轴是灰度值（一般0~255），
y轴就是图像中每一个灰度级对应的像素点的个数。

那么如何获得图像的直方图？首先来了解绘制直方图需要的一些量：
灰度级，正常情况下就是0-255共256个灰度级，从最黑一直到最亮（白）
（也有可能统计其中的某部分灰度范围），那么每一个灰度级对应一个数来储存该灰度对应的点数目。
也就是说直方图其实就是一个1*m（灰度级）的一个数组而已。
但是有的时候我们不希望一个一个灰度的递增，比如现在我想15个灰度一起作为一个灰度级来花直方图，
这个时候我们可能只需要1*(m/15)这样一个数组就够了。那么这里的15就是直方图的间隔宽度了。

Opencv给我们提供的函数是cv2.calcHist()，该函数有5个参数：

    image输入图像，传入时应该用中括号[]括起来
    channels:：传入图像的通道，如果是灰度图像，那就不用说了，只有一个通道，值为0，如果是彩色图像（有3个通道），那么值为0,1,2,中选择一个，对应着BGR各个通道。这个值也得用[]传入。
    mask：掩膜图像。如果统计整幅图，那么为none。主要是如果要统计部分图的直方图，就得构造相应的炎掩膜来计算。
    histSize：灰度级的个数，需要中括号，比如[256]
    ranges:像素值的范围，通常[0,256]，有的图像如果不是0-256，比如说你来回各种变换导致像素值负值、很大，则需要调整后才可以。

"""
# 画出图像的直方图
def hist_image(image):
    color = ("blue", "green", "red")
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


#直方图均衡化：提升对比度的两种方法：默认、自定义

#提升对比度（默认提升），只能是灰度图像
def equalHist_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("old", gray)#因为只能处理灰度图像，所以输出原图的灰度图像用于对比
    dst = cv.equalizeHist(gray)
    cv.imshow("normal", dst)

# 对比度限制（自定义提示参数）
def clahe_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # clipLimit是对比度的大小，tileGridSize是每次处理块的大小
    dst = clahe.apply(gray)
    cv.imshow("custorm", dst)


"""
颜色空间转换函数
cvtColor(src, code, dst=None, dstCn=None)  
opencv中有多种色彩空间，包括 RGB、HSI、HSL、HSV、HSB、YCrCb、CIE XYZ、CIE Lab8种，
使用中经常要遇到色彩空间的转化，以便生成mask图等操作


直方图均衡化

1、equalizeHist(src, dst=None)
图像的直方图是对图像对比度效果上的一种处理，旨在使得图像整体效果均匀，
黑与白之间的各个像素级之间的点更均匀一点。
直方图均衡化只要包括三个步骤：
    统计直方图中每个灰度级出现的次数；
    计算累计归一化直方图；
    重新计算像素点的像素值
    
2、cv2.normalize(src[, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]]) → dst
src-输入数组
dst-输出数组，支持原地运算
alpha-range normalization模式的最小值
beta-range normalization模式的最大值，不用于norm normalization(范数归一化)模式。
normType-归一化的类型，可以有以下的取值：

        NORM_MINMAX:数组的数值被平移或缩放到一个指定的范围，线性归一化，一般较常用。
        NORM_INF: 此类型的定义没有查到，根据OpenCV 1的对应项，可能是归一化数组的C-范数(绝对值的最大值)
        NORM_L1 :  归一化数组的L1-范数(绝对值的和)
        NORM_L2: 归一化数组的(欧几里德)L2-范数

dtype-dtype为负数时，输出数组的type与输入数组的type相同；否则，输出数组与输入数组只是通道数相同，而tpye=CV_MAT_DEPTH(dtype).
mask-操作掩膜，用于指示函数是否仅仅对指定的元素进行操作。
适用于：灰度级主要在0~150之间，造成图像对比度较低，可用直方图归一化将图像灰度级拉伸到0~255,使其更清晰。


限制对比度自适应直方图均衡化 
   
在进行完直方图均衡化之后，图片背景的对比度被改变了,因此会丢失了很多信息。
造成这种结果的根本原因在于图像的直方图并不是集中在某一个区域。
为了解决这个问题，我们需要使用自适应的直方图均衡化。

这种情况下，整幅图像会被分成很多小块，这些小块被称为“tiles”（在 OpenCV 中 tiles 的大小默认是 8x8），
然后再对每一个小块分别进行直方图均衡化（跟前面类似）。

所以在每一个的区域中，直方图会集中在某一个小的区域中（除非有噪声干扰）。
如果有噪声的话，噪声会被放大。为了避免这种情况的出现要使用对比度限制。
对于每个小块来说，如果直方图中的 bin 超过对比度的上限的话，
就把其中的像素点均匀分散到其他 bins 中，然后在进行直方图均衡化。
最后，为了去除每一个小块之间“人造的”（由于算法造成）边界，再使用双线性差值，对小块进行缝合。
createCLAHE(clipLimit=None, tileGridSize=None)
创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
"""

src = cv.imread("../resource/images/tiger.jpg")
# cv.imshow("old", src)
# hist_image(src)
equalHist_image(src)
clahe_image(src)
cv.waitKey(0)
cv.destroyAllWindows()
