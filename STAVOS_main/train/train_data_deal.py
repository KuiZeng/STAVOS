import cv2
import numpy as np           # 导入numpy库，用于处理和操作数组数据
from PIL import Image        # 导入Python Imaging Library，用于图像处理
import random                # 导入Python的随机数生成模块
import math                  # 导入Python的数学库

import warnings
warnings.filterwarnings('ignore')


def load_image_in_PIL(path, mode):
    img = Image.open(path)            # 打开指定路径下的文件
    img.load()                        # 为了加速图像处理，提前解码整个图像
    return img.convert(mode)          # 将图像转换为指定模式（灰度或RGB）


# 接受5个参数：旋转角度（degree）、平移距离（translate）、缩放范围（scale_ranges）、错切程度（shear）和图像大小（img_size）。
def random_affine_params(degree, translate, scale_ranges, shear, img_size):
    angle = random.uniform(-degree, degree)     # 生成一个随机浮点数，其值在[-degree, degree]区间内，表示旋转的角度。
    max_dx = translate * img_size[0]            # 计算可以进行的最大水平和垂直平移距离，然后生成两个随机浮点数，其值在[-max_dx, max_dx]和[-max_dy, max_dy]区间内，分别表示水平和垂直方向上的平移距离，并四舍五入取整。
    max_dy = translate * img_size[1]
    translations = (np.round(random.uniform(-max_dx, max_dx)),
                    np.round(random.uniform(-max_dy, max_dy)))
    scale = random.uniform(scale_ranges[0], scale_ranges[1])        # 生成一个随机浮点数，其值在[scale_ranges[0], scale_ranges[1]]区间内，表示缩放比例。
    shear = [random.uniform(-shear, shear), random.uniform(-shear, shear)]  # 生成两个随机浮点数，其值在[-shear, shear]区间内，分别表示在x轴和y轴上的错切程度。
    return angle, translations, scale, shear    # 返回四个值，即对应旋转角度、平移距离、缩放比例和错切程度。


# 定义函数，输入为图片、缩放比例和纵横比，返回随机裁剪的参数
def random_crop_params(img, scale, ratio=(3 / 4, 4 / 3)):
    # 获取图片的尺寸
    width, height = img.size
    # 计算图片面积
    area = height * width

    # 循环10次来随机选取裁剪区域
    for attempt in range(10):
        # 随机选取目标区域面积
        target_area = random.uniform(*scale) * area
        # 取对数的纵横比度量
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        # 随机选取宽高比
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        # 根据选定的面积和纵横比计算出裁剪后的宽和高
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        # 如果宽和高在图片范围之内，则随机选取一个起点并返回参数
        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # 如果循环10次都没有找到合适的裁剪区域，则按照纵横比调整裁剪区域
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def getEdgeClear(img, mask, x):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.float64)
    mask[mask > 0] = 255

    # sobel算子
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # cv2.imwrite("img_sobel.png", img_sobel)

    # 创建一个5x5的核，用于腐蚀和膨胀操作
    kernel = np.ones((x, x), np.uint8)

    # 腐蚀操作，向内扩展x像素
    mask_inner = cv2.erode(mask, kernel, iterations=1)
    # cv2.imwrite("mask_inner.png", mask_inner)

    # 膨胀操作，向外扩展x像素
    mask_outer = cv2.dilate(mask, kernel, iterations=1)
    # cv2.imwrite("mask_outer.png", mask_outer)

    # 从向外扩展的掩膜中减去向内扩展的掩膜，得到环状区域
    mask_roi = cv2.subtract(mask_outer, mask_inner)
    # cv2.imwrite("mask_roi.png", mask_roi)

    # 求ROI的清晰度
    # roi_sobel = np.where(mask_roi == 255, img_sobel, mask_roi)  # img_gray[(mask_roi != 255)] = [0]
    # cv2.imwrite("roi_sobel.png", roi_sobel)

    # 只求交叉区域的清晰度
    clear_list = img_sobel[(mask_roi == 255)]
    if len(clear_list) == 0:
        return 0
    else:
        return sum(clear_list) / len(clear_list)