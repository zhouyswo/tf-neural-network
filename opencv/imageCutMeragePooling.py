import cv2 as cv
import numpy as np

img = cv.imread("../resource/images/tiger.jpg")
#截取第5行到89行的第500列到630列的区域
# cut = img[160:450,300:600]
# cv.imshow("cut",cut)
#指定位置填充，大小要一样才能填充
# img[160:450,700:1000]=cut
# cv.imshow("img",img)

copyImg= img.copy()
h, w = img.shape[:2]  # 读取图像的宽和高
mask = np.zeros([h + 2, w + 2], np.uint8)  # 新建图像矩阵  +2是官方函数要求
cv.floodFill(copyImg, mask, (0, 100), (0, 100, 255), (100, 100, 50), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
cv.imshow("copyImg",copyImg)

cv.waitKey(0)
cv.destroyAllWindows()

"""
漫水填充法
floodFill(image, mask, seedPoint, newVal, loDiff=None, upDiff=None, flags=None)
第一个参数，InputOutputArray类型的image, 输入/输出1通道或3通道，8位或浮点图像，具体参数由之后的参数具体指明。
第二个参数， InputOutputArray类型的mask，这是第二个版本的floodFill独享的参数，表示操作掩模,。它应该为单通道、8位、长和宽上都比输入图像 image 大两个像素点的图像。第二个版本的floodFill需要使用以及更新掩膜，所以这个mask参数我们一定要将其准备好并填在此处。需要注意的是，漫水填充不会填充掩膜mask的非零像素区域。例如，一个边缘检测算子的输出可以用来作为掩膜，以防止填充到边缘。同样的，也可以在多次的函数调用中使用同一个掩膜，以保证填充的区域不会重叠。另外需要注意的是，掩膜mask会比需填充的图像大，所以 mask 中与输入图像(x,y)像素点相对应的点的坐标为(x+1,y+1)。
第三个参数，Point类型的seedPoint，漫水填充算法的起始点。
第四个参数，Scalar类型的newVal，像素点被染色的值，即在重绘区域像素的新值。
第5参数，Scalar类型的loDiff，有默认值Scalar( )，表示当前观察像素值与其部件邻域像素值或者待加入该部件的种子像素之间的亮度或颜色之负差（lower brightness/color difference）的最大值。 
第6个参数，Scalar类型的upDiff，有默认值Scalar( )，表示当前观察像素值与其部件邻域像素值或者待加入该部件的种子像素之间的亮度或颜色之正差（lower brightness/color difference）的最大值。
第7个参数，int类型的flags，操作标志符，此参数包含三个部分，比较复杂，我们一起详细看看。
            低八位（第0~7位）用于控制算法的连通性，可取4 (4为缺省值) 或者 8。如果设为4，表示填充算法只考虑当前像素水平方向和垂直方向的相邻点；如果设为 8，除上述相邻点外，还会包含对角线方向的相邻点。
            高八位部分（16~23位）可以为0 或者如下两种选项标识符的组合：     
            FLOODFILL_FIXED_RANGE - 如果设置为这个标识符的话，就会考虑当前像素与种子像素之间的差，否则就考虑当前像素与其相邻像素的差。也就是说，这个范围是浮动的。
            FLOODFILL_MASK_ONLY - 如果设置为这个标识符的话，函数不会去填充改变原始图像 (也就是忽略第三个参数newVal), 而是去填充掩模图像（mask）。这个标识符只对第二个版本的floodFill有用，因第一个版本里面压根就没有mask参数。
"""
